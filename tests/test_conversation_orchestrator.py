import io
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from conversation_orchestrator import (
    ConversationOrchestrator,
    JsonHttpClient,
    OrchestratorError,
    OrchestratorSettings,
    PersistenceError,
    SYSTEM_PROMPT,
    build_chat_prompt,
    message_tokens,
    parse_reply,
)
from scripts.run_chat_demo import handle_command, main as demo_main


def context_payload():
    return {
        "as_of": "2026-09-14T08:00:00+00:00",
        "profile": {"routine": {"tennis_time": "Cumartesi sabah saat 7'de tenis oynarım"}},
        "session": {},
        "temporary_observations": [],
        "recent_messages": [
            {"role": "user", "content": "Merhaba"},
            {"role": "assistant", "content": "Size nasıl yardımcı olabilirim?"},
        ],
        "memory_refs": ["fact-1"],
        "profile_retrieval": {"pinned_categories": ["communication"]},
    }


def reply_payload():
    return {
        "message": {"role": "assistant", "content": "Cumartesi sabah saat 7'de."},
        "done": True, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 12,
    }


def decision_payload():
    return {"status": "processed", "decisions": [{
        "candidate_id": "candidate-1", "memory_type": "discard", "status": "ignored",
        "analyzer_source": "ollama",
    }]}


def attributed_context_payload():
    context = context_payload()
    context["user_id"] = "test-user"
    context["profile_facts"] = [{
        "fact_id": "coffee-1", "owner_user_id": "test-user",
        "category": "routine", "key": "morning_coffee",
        "value": "Her sabah bol köpüklü Türk kahvesi içerim",
        "provenance": {
            "source_event_id": "event-1", "verification_status": "unverified",
            "confidence": 0.95,
        },
    }]
    return context


def emergency_context_payload():
    context = attributed_context_payload()
    context["profile_facts"] = [{
        "fact_id": "emergency-1", "owner_user_id": "test-user",
        "category": "emergency_contact", "key": "primary_emergency_contact",
        "value": "Acil durumda kızım Ayşe'yi 0555 123 45 67 numarasından ara",
        "provenance": {
            "source_event_id": "event-emergency",
            "verification_status": "user_confirmed",
            "confidence": 1.0,
        },
    }]
    return context


class ConversationOrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.memory = Mock(spec=JsonHttpClient)
        self.ollama = Mock(spec=JsonHttpClient)
        self.memory.request.side_effect = [context_payload(), decision_payload(), {"status": "created"}]
        self.ollama.request.return_value = reply_payload()
        self.orchestrator = ConversationOrchestrator(
            OrchestratorSettings(async_memory_ingestion=False),
            "test-user", "test-session",
            memory_http=self.memory, ollama_http=self.ollama,
        )

    def test_turn_order_uses_memory_and_saves_user_and_assistant_once(self):
        parent = Mock()
        parent.attach_mock(self.memory, "memory")
        parent.attach_mock(self.ollama, "ollama")
        turn = self.orchestrator.chat("Tenise ne zaman giderim?")

        paths = [call.args[1] for call in parent.mock_calls]
        self.assertEqual(paths, [
            "/v1/context:build", "/api/chat", "/v1/interactions:process",
            "/v1/sessions/test-session/messages",
        ])
        query_payload = self.memory.request.call_args_list[0].args[2]
        self.assertEqual(query_payload["query"], turn.text)
        messages = self.ollama.request.call_args.args[2]["messages"]
        self.assertNotIn("format", self.ollama.request.call_args.args[2])
        self.assertEqual(turn.model_stats["response_mode"], "native_ollama_chat")
        self.assertIn("Cumartesi sabah saat 7", messages[0]["content"])
        self.assertNotIn("memory_refs", messages[0]["content"])
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(turn.prompt.history_message_count, 2)
        self.assertEqual(sum(message["content"] == turn.text for message in messages), 1)
        user_write = self.memory.request.call_args_list[1].args[2]
        self.assertEqual(user_write["event_id"], turn.event_id)
        self.assertNotIn("recent_messages", user_write)
        assistant_write = self.memory.request.call_args_list[2].args[2]
        self.assertEqual(assistant_write["message_id"], turn.assistant_message_id)
        self.assertEqual(assistant_write["role"], "assistant")
        self.assertEqual(assistant_write["content"], turn.reply)
        self.assertEqual(assistant_write["parent_message_id"], turn.event_id)
        self.assertLessEqual(turn.user_occurred_at, turn.assistant_occurred_at)
        self.assertIsNone(self.orchestrator.pending_turn)

    def test_emergency_bypasses_reply_model_but_keeps_memory_persistence(self):
        self.memory.request.side_effect = [
            emergency_context_payload(), decision_payload(), {"status": "created"},
        ]
        self.ollama.reset_mock()

        turn = self.orchestrator.chat("Banyoda düştüm, her yer kan oldu")

        self.ollama.request.assert_not_called()
        self.assertEqual(turn.model_stats["response_mode"], "emergency_orchestrator")
        self.assertEqual(turn.emergency_action.action, "call_simulated")
        self.assertEqual(turn.emergency_action.contact_name, "Ayşe")
        self.assertIn("gerçek arama yapmaz", turn.reply)
        self.assertEqual(
            [call.args[1] for call in self.memory.request.call_args_list],
            [
                "/v1/context:build",
                "/v1/interactions:process",
                "/v1/sessions/test-session/messages",
            ],
        )

    def test_confirmation_commands_are_removed(self):
        self.memory.request.side_effect = None
        for command in ("/confirm candidate-1", "/reject candidate-1", "/pending"):
            with self.subTest(command=command), patch("sys.stdout", new=io.StringIO()):
                self.assertTrue(handle_command(self.orchestrator, command, False))
                self.memory.request.assert_not_called()
        self.ollama.request.assert_not_called()

    def test_generation_failure_does_not_write_memory(self):
        self.ollama.request.side_effect = OrchestratorError("Ollama kapalı")
        with self.assertRaises(OrchestratorError):
            self.orchestrator.chat("Merhaba")
        self.assertEqual(self.memory.request.call_count, 1)
        self.assertIsNone(self.orchestrator.pending_turn)

    def test_empty_or_unfinished_generation_does_not_write_memory(self):
        for payload in [
            {"message": {"content": ""}},
            {"message": {"content": "Cevap"}, "done": False},
            {"message": {"role": "user", "content": "Cevap"}},
            {"message": {"content": "x" * 10_001}},
        ]:
            with self.subTest(payload=str(payload)[:60]):
                self.memory.request.reset_mock()
                self.memory.request.side_effect = [context_payload()]
                self.ollama.request.return_value = payload
                with self.assertRaises(OrchestratorError):
                    self.orchestrator.chat("Merhaba")
                self.assertEqual(self.memory.request.call_count, 1)

    def test_interaction_failure_retry_keeps_ids_and_does_not_regenerate(self):
        self.memory.request.side_effect = [context_payload(), OrchestratorError("timeout")]
        with self.assertRaises(PersistenceError):
            self.orchestrator.chat("Merhaba")
        failed_payload = self.memory.request.call_args.args[2]
        with self.assertRaises(OrchestratorError):
            self.orchestrator.chat("Başka mesaj")
        self.memory.request.side_effect = [decision_payload(), {"status": "created"}]
        turn = self.orchestrator.retry_pending()
        self.assertEqual(self.memory.request.call_args_list[-2].args[2], failed_payload)
        self.assertEqual(turn.event_id, failed_payload["event_id"])
        self.ollama.request.assert_called_once()
        self.assertIsNone(self.orchestrator.pending_turn)

    def test_assistant_failure_retries_only_assistant_write_with_same_id(self):
        self.memory.request.side_effect = [context_payload(), decision_payload(), OrchestratorError("timeout")]
        with self.assertRaises(PersistenceError):
            self.orchestrator.chat("Merhaba")
        failed_payload = self.memory.request.call_args.args[2]
        self.memory.request.side_effect = [{"status": "duplicate"}]
        self.orchestrator.retry_pending()
        self.assertEqual(self.memory.request.call_args.args[2], failed_payload)
        self.assertEqual(sum(call.args[1] == "/v1/interactions:process" for call in self.memory.request.call_args_list), 1)
        self.ollama.request.assert_called_once()

    def test_healthcheck_requires_installed_response_model(self):
        self.memory.request.side_effect = [{"status": "ok"}, {"provider": "ollama", "available": True}]
        self.ollama.request.return_value = {"models": []}
        with self.assertRaisesRegex(OrchestratorError, "ollama pull"):
            self.orchestrator.check_services()

    def test_chat_configuration_does_not_change_extractor_configuration(self):
        with patch.dict(os.environ, {
            "OLLAMA_MODEL": "qwen3:8b", "CHAT_OLLAMA_MODEL": "response-model",
            "CHAT_OLLAMA_NUM_CTX": "2048", "CHAT_OLLAMA_NUM_PREDICT": "256",
        }, clear=True):
            settings = OrchestratorSettings.from_environment()
            self.assertEqual(settings.model, "response-model")
            self.assertEqual(os.environ["OLLAMA_MODEL"], "qwen3:8b")
            self.assertEqual(settings.num_ctx, 2048)

    def test_async_memory_ingestion_is_the_default_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                OrchestratorSettings.from_environment().async_memory_ingestion
            )
        with patch.dict(
            os.environ,
            {"MEMORY_ASYNC_INGESTION": "false"},
            clear=True,
        ):
            self.assertFalse(
                OrchestratorSettings.from_environment().async_memory_ingestion
            )

    def test_generation_options_are_shared_defaults_but_fresh_objects(self):
        settings = OrchestratorSettings(model="qwen3:8b", num_ctx=2048, num_predict=256)
        expected = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                    "num_ctx": 2048, "num_predict": 256}
        first = settings.generation_options
        self.assertEqual(first, expected)
        first["temperature"] = 99
        self.assertEqual(settings.generation_options, expected)
        self.orchestrator.chat("Merhaba")
        self.assertEqual(self.ollama.request.call_args.args[2]["options"],
                         self.orchestrator.settings.generation_options)

    def test_gemma4_sampling_uses_conservative_demo_settings_and_fresh_objects(self):
        expected = {"temperature": 0.3, "top_p": 0.95, "top_k": 64, "min_p": 0,
                    "num_ctx": 2048, "num_predict": 256}
        for model in ("gemma4", "gemma4:12b", "gemma4:latest", "gemma4:e4b"):
            with self.subTest(model=model):
                settings = OrchestratorSettings(model=model, num_ctx=2048, num_predict=256)
                first = settings.generation_options
                self.assertEqual(first, expected)
                first["temperature"] = 99
                self.assertEqual(settings.generation_options, expected)

    def test_gemma4_sampling_does_not_match_substrings_or_other_families(self):
        expected = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                    "num_ctx": 4096, "num_predict": 512}
        for model in ("qwen3:8b", "gemma3:12b", "gemma4-custom:12b",
                      "my-gemma4:12b", "unknown"):
            with self.subTest(model=model):
                self.assertEqual(OrchestratorSettings(model=model).generation_options, expected)

    def test_gemma4_chat_uses_reply_sampling_without_changing_extractor(self):
        settings = OrchestratorSettings(
            model="gemma4:12b",
            async_memory_ingestion=False,
        )
        orchestrator = ConversationOrchestrator(
            settings, "test-user", "test-session",
            memory_http=self.memory, ollama_http=self.ollama,
        )
        orchestrator.chat("Merhaba")
        request = self.ollama.request.call_args.args[2]
        self.assertEqual(request["model"], "gemma4:12b")
        self.assertEqual(request["options"], settings.generation_options)
        self.assertEqual(OrchestratorSettings(model="qwen3:8b").generation_options["temperature"], 0.7)

    def test_only_real_history_and_current_text_are_sent_and_persisted(self):
        turn = self.orchestrator.chat("Bugün ne konuşalım?")
        prompt = turn.prompt
        self.assertEqual(prompt.history_message_count, 2)
        self.assertEqual(prompt.estimated_tokens,
                         sum(message_tokens(message) for message in prompt.messages))
        self.assertLessEqual(prompt.estimated_tokens, prompt.input_budget)
        self.assertTrue(prompt.messages[0]["content"].startswith(SYSTEM_PROMPT))
        self.assertNotIn('assistant: {"answer"', prompt.messages[0]["content"])
        self.assertNotIn("ÜSLUP ÖRNEKLERİ", prompt.messages[0]["content"])
        self.assertEqual(prompt.messages[1:-1], context_payload()["recent_messages"])
        for message in prompt.messages[1:]:
            self.assertNotIn(SYSTEM_PROMPT, message["content"])
        writes = self.memory.request.call_args_list[1:]
        self.assertEqual(len(writes), 2)
        self.assertEqual(writes[0].args[2]["text"], turn.text)
        self.assertEqual(writes[1].args[2]["content"], turn.reply)
        self.assertNotIn(SYSTEM_PROMPT, writes[0].args[2]["text"])
        self.assertNotIn(SYSTEM_PROMPT, writes[1].args[2]["content"])

    def test_style_and_system_do_not_hardcode_the_reported_personal_message(self):
        instructions = SYSTEM_PROMPT
        for reported_term in ["kuzen", "ayşe", "pencere", "balkon", "hava", "çorba"]:
            self.assertNotIn(reported_term, instructions.casefold())

    def test_declaration_uses_arbitrary_native_model_answer_unchanged(self):
        # This only proves transport/persistence, not the model's conversation quality.
        model_answer = "Bilgiyi paylaştığınız için teşekkür ederim."
        response = reply_payload()
        response["message"]["content"] = model_answer
        self.ollama.request.return_value = response
        turn = self.orchestrator.chat("Komşumun adı Zeynep.")
        self.assertEqual(turn.reply, model_answer)
        self.assertEqual(self.ollama.request.call_args.args[2]["messages"][-1],
                         {"role": "user", "content": "Komşumun adı Zeynep."})
        self.assertEqual(self.memory.request.call_args_list[1].args[2]["text"], turn.text)
        self.assertEqual(self.memory.request.call_args_list[2].args[2]["content"], model_answer)

    def test_database_fact_is_attributed_and_not_duplicated_as_legacy_profile(self):
        context = attributed_context_payload()
        context["profile_facts"][0]["provenance"]["source_quote"] = "Unrelated protected text"
        original = json.dumps(context)
        prompt = build_chat_prompt(context, "Sabahları ne içerim?", OrchestratorSettings(), user_id="test-user")
        data = json.loads(prompt.messages[0]["content"].split("\nMEMORY_CONTEXT_JSON:\n", 1)[1])
        self.assertEqual(data["user_id"], "test-user")
        self.assertEqual(data["profile_facts"][0]["owner_user_id"], "test-user")
        self.assertEqual(data["profile_facts"][0]["value"], context["profile_facts"][0]["value"])
        self.assertEqual(data["profile_facts"][0]["provenance"]["source_event_id"], "event-1")
        self.assertNotIn("profile", data)
        self.assertNotIn("Unrelated protected text", prompt.messages[0]["content"])
        self.assertEqual(prompt.profile_fact_count, 1)
        self.assertEqual(json.dumps(context), original)

    def test_legacy_profile_is_attributed_without_fabricated_source_or_normalization(self):
        context = context_payload()
        prompt = build_chat_prompt(context, "Saatim neydi?", OrchestratorSettings(), user_id="test-user")
        data = json.loads(prompt.messages[0]["content"].split("\nMEMORY_CONTEXT_JSON:\n", 1)[1])
        fact = data["profile_facts"][0]
        self.assertEqual(fact["value"], context["profile"]["routine"]["tennis_time"])
        self.assertEqual(fact["owner_user_id"], "test-user")
        self.assertNotIn("fact_id", fact)
        self.assertNotIn("provenance", fact)

    def test_invalid_or_other_user_facts_fail_before_model_and_persistence(self):
        contexts = []
        wrong_context = attributed_context_payload()
        wrong_context["user_id"] = "other-user"
        contexts.append(wrong_context)
        wrong_fact = attributed_context_payload()
        wrong_fact["profile_facts"][0]["owner_user_id"] = "other-user"
        contexts.append(wrong_fact)
        malformed = attributed_context_payload()
        malformed["profile_facts"] = None
        contexts.append(malformed)
        for context in contexts:
            with self.subTest(context=context):
                self.memory.request.reset_mock()
                self.memory.request.side_effect = [context]
                self.ollama.request.reset_mock()
                with self.assertRaises(OrchestratorError):
                    self.orchestrator.chat("Sabahları ne içerim?")
                self.ollama.request.assert_not_called()
                self.assertEqual(self.memory.request.call_count, 1)
                self.assertIsNone(self.orchestrator.pending_turn)

    def test_attributed_fact_envelope_is_fully_budgeted_and_trimmed_as_one_unit(self):
        context = attributed_context_payload()
        context["profile_facts"].append({
            **context["profile_facts"][0], "fact_id": "large", "key": "large",
            "value": {"nested": ["x" * 20_000]},
        })
        original = json.dumps(context)
        prompt = build_chat_prompt(context, "Merhaba", OrchestratorSettings(num_ctx=2048, num_predict=256))
        data = json.loads(prompt.messages[0]["content"].split("\nMEMORY_CONTEXT_JSON:\n", 1)[1])
        self.assertTrue(prompt.trimmed)
        self.assertEqual([fact["fact_id"] for fact in data["profile_facts"]], ["coffee-1"])
        self.assertEqual(prompt.profile_fact_count, len(data["profile_facts"]))
        self.assertLessEqual(prompt.estimated_tokens, prompt.input_budget)
        self.assertEqual(prompt.estimated_tokens, sum(message_tokens(message) for message in prompt.messages))
        self.assertEqual(json.dumps(context), original)

    def test_final_sentence_comes_from_model_not_database_formatter(self):
        context = attributed_context_payload()
        self.memory.request.side_effect = [context, decision_payload(), {"status": "created"}]
        model_answer = "Sabahları bol köpüklü Türk kahvesi içiyorsunuz."
        response = reply_payload()
        response["message"]["content"] = model_answer
        self.ollama.request.return_value = response
        turn = self.orchestrator.chat("Sabahları ne içerim?")
        self.assertEqual(turn.reply, model_answer)
        self.assertNotEqual(turn.reply, context["profile_facts"][0]["value"])
        self.assertEqual(self.memory.request.call_args_list[2].args[2]["content"], model_answer)

    def test_prompt_globally_bounds_memory_history_and_current_turn(self):
        context = context_payload()
        context["profile"] = {
            "communication": {"form_of_address": "Ahmet Bey"},
            "routine": {"very_large_fact": "x" * 20_000},
        }
        context["recent_messages"] = [{"role": "user", "content": "h" * 10_000}] * 10
        context["temporary_observations"] = [{"value": "o" * 10_000}]
        original = json.dumps(context)
        prompt = build_chat_prompt(context, "Merhaba", OrchestratorSettings(num_ctx=2048, num_predict=256))
        self.assertLessEqual(prompt.estimated_tokens, prompt.input_budget)
        self.assertIn("Ahmet Bey", prompt.messages[0]["content"])
        self.assertNotIn("very_large_fact", prompt.messages[0]["content"])
        self.assertEqual(prompt.messages[-1], {"role": "user", "content": "Merhaba"})
        self.assertTrue(prompt.trimmed)
        self.assertEqual(json.dumps(context), original)

    def test_oversized_current_text_is_rejected_not_silently_truncated(self):
        with self.assertRaisesRegex(OrchestratorError, "context bütçesine"):
            build_chat_prompt({}, "x" * 9000, OrchestratorSettings(num_ctx=2048, num_predict=256))

    def test_reply_parser_validates_text_not_language_or_numbers(self):
        # A transport unit test does not establish semantic accuracy. Wrong
        # model answers must stay visible for evaluation, not become 'no memory'.
        for answer in ["Dört.", "Saat 15:00'da.", "Ahmet Bey, bugün nasıl hissediyorsunuz?", "  Bilmiyorum.  "]:
            with self.subTest(answer=answer):
                self.assertEqual(parse_reply(answer), answer.strip())

    def test_reply_parser_rejects_empty_oversized_or_non_text_content(self):
        for payload in [
            None, {}, [], 7, True, "", " \n\t ", "x" * 10_001,
        ]:
            with self.subTest(payload=str(payload)[:60]), self.assertRaises(OrchestratorError):
                parse_reply(payload)

    def test_json_looking_native_text_is_not_decoded_or_rewritten(self):
        content = '{"answer":"Literal model text","extra":true}'
        self.assertEqual(parse_reply(content), content)
        self.ollama.request.return_value = {
            "message": {"role": "assistant", "content": content}, "done": True,
        }
        turn = self.orchestrator.chat("Merhaba")
        self.assertEqual(turn.reply, content)
        self.assertEqual(self.memory.request.call_args.args[2]["content"], content)

    def test_invalid_native_text_never_writes_memory(self):
        for content in [None, [], {}, 7, "", " \n\t ", "x" * 10_001]:
            with self.subTest(content=str(content)[:30]):
                self.memory.request.reset_mock()
                self.memory.request.side_effect = [context_payload()]
                self.ollama.request.return_value = {
                    "message": {"role": "assistant", "content": content}, "done": True,
                }
                with self.assertRaises(OrchestratorError):
                    self.orchestrator.chat("Merhaba")
                self.assertEqual(self.memory.request.call_count, 1)
                self.assertIsNone(self.orchestrator.pending_turn)

    def test_advice_with_no_profile_is_not_replaced_and_is_saved_unchanged(self):
        self.memory.request.side_effect = [{}, decision_payload(), {"status": "created"}]
        model_answer = "Bugün tenis oynamak istemenizin özel bir nedeni var mı?"
        response = reply_payload()
        response["message"]["content"] = model_answer
        self.ollama.request.return_value = response
        turn = self.orchestrator.chat("Tenis oynayayım mı?")
        self.assertEqual(turn.reply, model_answer)
        self.assertEqual(self.memory.request.call_args.args[2]["content"], model_answer)

    def test_discarded_memory_does_not_discard_conversation_answer(self):
        turn = self.orchestrator.chat("İki artı iki kaç eder?")
        self.assertEqual(turn.decisions[0]["memory_type"], "discard")
        self.assertEqual(turn.reply, reply_payload()["message"]["content"])

    def test_memory_failure_keeps_generated_answer_for_explicit_retry(self):
        self.memory.request.side_effect = [context_payload(), OrchestratorError("analyzer unavailable")]
        with self.assertRaises(PersistenceError):
            self.orchestrator.chat("Tenis oynayayım mı?")
        turn = self.orchestrator.pending_turn
        self.assertEqual(turn.reply, reply_payload()["message"]["content"])
        self.ollama.request.assert_called_once()

    def test_native_plain_text_answer_is_persisted_without_a_template(self):
        answer = "Pencereyi açmak istiyorsunuz, hava nasıl?"
        self.ollama.request.return_value = {
            "message": {"role": "assistant", "content": answer}, "done": True,
        }
        turn = self.orchestrator.chat("06 - Pencereyi açmak istiyorum.")
        self.assertEqual(turn.reply, answer)
        self.assertEqual(self.memory.request.call_args.args[2]["content"], answer)
        self.assertEqual(self.ollama.request.call_args.args[2]["messages"][-1],
                         {"role": "user", "content": "06 - Pencereyi açmak istiyorum."})

    def test_question_and_assistant_history_are_passed_without_regex_routing(self):
        context = context_payload()
        context["recent_messages"] = [
            {"role": "user", "content": "Her sabah kaçta tenis oynarım"},
            {"role": "assistant", "content": "Her gün 15:00'da oynarsınız."},
        ]
        prompt = build_chat_prompt(context, "Saatim neydi?", OrchestratorSettings())
        self.assertEqual(prompt.messages[1:-1], context["recent_messages"])
        self.assertNotIn("source_refs", prompt.messages[0]["content"])
        self.assertNotIn("memory_refs", prompt.messages[0]["content"])

    def test_single_word_turns_use_model_reply_without_copying_address_preference(self):
        for text, model_answer in [
            ("tenis", "Ahmet Bey, tenis hakkında nasıl yardımcı olabilirim?"),
            ("Emekliyim", "Anladım, Ahmet Bey."),
        ]:
            with self.subTest(text=text):
                context = context_payload()
                context["profile"]["communication"] = {"form_of_address": "Bana Ahmet Bey diye hitap et"}
                self.memory.request.side_effect = [context, decision_payload(), {"status": "created"}]
                self.ollama.request.reset_mock()
                response = reply_payload()
                response["message"]["content"] = model_answer
                self.ollama.request.return_value = response

                turn = self.orchestrator.chat(text)

                self.ollama.request.assert_called_once()
                method, path, request = self.ollama.request.call_args.args
                self.assertEqual((method, path), ("POST", "/api/chat"))
                self.assertEqual(request["messages"][-1], {"role": "user", "content": text})
                self.assertIn("Bana Ahmet Bey diye hitap et", request["messages"][0]["content"])
                self.assertEqual(turn.reply, model_answer)
                self.assertNotIn("Bana Ahmet Bey diye hitap et", turn.reply)
                self.assertEqual(turn.model_stats["done_reason"], "stop")
                self.assertEqual(turn.model_stats["prompt_eval_count"], 300)
                self.assertEqual(turn.model_stats["eval_count"], 12)
                self.assertEqual(self.memory.request.call_args.args[2]["content"], model_answer)

    def test_one_shot_cli_processes_multiple_turns(self):
        with patch("scripts.run_chat_demo.ConversationOrchestrator") as factory, patch("sys.stdout", new=io.StringIO()), patch.dict(os.environ, {}, clear=True):
            demo = factory.return_value
            demo.pending_turn = None
            demo.check_services.return_value = {"provider": "ollama", "model": "qwen3:8b", "available": True}
            demo.chat.side_effect = self.orchestrator.chat
            self.memory.request.side_effect = [
                context_payload(), decision_payload(), {"status": "created"},
                context_payload(), decision_payload(), {"status": "created"},
            ]
            exit_code = demo_main(["--user-id", "test-user", "--text", "Merhaba", "--text", "Tenise ne zaman giderim?"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(demo.chat.call_count, 2)


class JsonHttpClientTest(unittest.TestCase):
    def test_transport_sends_unicode_json_and_parses_response(self):
        response = Mock()
        response.read.return_value = '{"status":"ok"}'.encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch("conversation_orchestrator.urllib_request.urlopen", return_value=response) as urlopen:
            result = JsonHttpClient("http://127.0.0.1:8001", 120).request("POST", "/test", {"text": "Çay"})
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(json.loads(urlopen.call_args.args[0].data), {"text": "Çay"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)

    def test_http_error_does_not_echo_sensitive_body(self):
        failure = HTTPError("http://localhost/test", 422, "error", {}, io.BytesIO(b"secret-health-data"))
        with patch("conversation_orchestrator.urllib_request.urlopen", side_effect=failure):
            with self.assertRaises(OrchestratorError) as caught:
                JsonHttpClient("http://localhost", 120).request("POST", "/test")
        self.assertIn("HTTP 422", str(caught.exception))
        self.assertNotIn("secret-health-data", str(caught.exception))

    def test_network_error_is_actionable(self):
        with patch("conversation_orchestrator.urllib_request.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(OrchestratorError, "erişilemiyor"):
                JsonHttpClient("http://localhost", 120).request("GET", "/test")
