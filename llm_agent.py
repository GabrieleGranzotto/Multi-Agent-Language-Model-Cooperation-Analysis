import os, time, re, json, ollama, random

class OllamaAgent:
    def __init__(self, game, model_name: str = "gemma4", debug: bool = False):
        self.model_name = model_name
        self.game = game
        self.messages = []
        self.id = random.randint(1000, 9999)  # Unique identifier for the agent
        self.sanity_retries = 0
        self.hardcoded_fallbacks = 0
        self.sanity_fail_reasons: dict[str, int] = {}
        self.debug = debug

    def handshake(self, opponent_agent: "OllamaAgent") -> None:
        """
        Retrieve the unique identifier for the opponent agent and initialize context.
        """
        self.opponent_id = opponent_agent.id
        self._context_definition()

    def _add_message(self, role: str, content: str, sender: str = "AGENT"):
        # Valid roles are "system", "user", and "assistant"
        if role == "system":
            tag = "SYSTEM"
        elif role == "assistant":
            tag = str(self.id)
        else:
            tag = "JUDGE" if sender == "JUDGE" else str(self.opponent_id)
        self.messages.append({"role": role, "content": f"[{tag}]: {content}"})

    def respond(self, json_mode: bool = False) -> str:
        """
        Send the conversation history to the Ollama model and record its reply.
        """
        content = None
        try:
            if json_mode:
                try:
                    response = ollama.chat(model=self.model_name, messages=self.messages, format="json")
                except Exception as e:
                    if getattr(e, "status_code", None) == 501 or "501" in str(e):
                        response = ollama.chat(model=self.model_name, messages=self.messages)
                    else:
                        raise RuntimeError(f"Failed to use JSON format with Ollama API: {e}")
            else:
                response = ollama.chat(model=self.model_name, messages=self.messages)
                cnt = 0
                agent_turns_so_far = sum(1 for m in self.messages if m["role"] == "assistant")
                while not self._sanity_check(response["message"]["content"]):
                    cnt += 1
                    if cnt > 3:
                        self.hardcoded_fallbacks += 1
                        if agent_turns_so_far < 2:
                            content = "Let's discuss our approach for this round so we can reach a mutually beneficial outcome."
                        else:
                            content = "I have stated my position clearly and I am ready to conclude our discussion. [DONE]"
                        break
                    self.sanity_retries += 1
                    response = ollama.chat(model=self.model_name, messages=self.messages)
        except Exception as e:
            raise RuntimeError(f"Failed to get response from Ollama API: {e}")

        if content is None:
            raw_text = response["message"]["content"]
            content = raw_text if json_mode else self._clean_candidate(raw_text)

        self._add_message("assistant", content)
        return content

    def listen(self, prompt: str, user: str = "AGENT"):
        """
        Listen for a user's input and add it to the conversation history.
        """
        if prompt is None:
            raise ValueError("Prompt cannot be None. Please provide a valid input.")
        self._add_message("user", prompt, sender=user)

    def _context_definition(self):
        """
        Define the context for the agent's behavior and capabilities.
        """
        context = f"""
You are an autonomous agent called [{self.id}] participating in a behavioral economics experiment, playing a one-shot classic social dilemma (Prisoner's Dilemma). Your opponent is another agent called [{self.opponent_id}].
Your goal is to secure the best possible outcome for yourself. To do so, you must carefully weigh the mathematical risks of the payoff matrix and evaluate how much you can actually trust the other agent based on your conversation.

Description of the game:
{self.game.description}

Current Phase — Pre-Game Negotiation:
You are chatting directly with the other participant, referred to as [{self.opponent_id}], to discuss the intentions of the game and negotiate a strategy before playing.
- Communicate naturally in plain text as [{self.opponent_id}]'s counterpart.
- Do NOT output JSON, final move declarations, or meta-commentary about the game ending or awaiting a judge (the [DONE] tag below is the only exception).
- Keep your messages focused on discussing trust, incentives, and your proposed strategy for the upcoming single round.
- NEVER include [DONE] in your opening (first) message; you must engage in at least one exchange before closing the discussion.
- After the opening exchange, when you are ready to conclude the negotiation and move to your final decision, end your message with [DONE].
- If [{self.opponent_id}] includes [DONE] and you have nothing further to negotiate, you MUST also end your reply with [DONE] to close the conversation; otherwise, omit [DONE] to continue discussing.
        """.strip()
        self._add_message("system", context)

    def _parse_json_response(self, raw_response: str) -> tuple[str, str]:
        """
        Extract and validate the JSON object from the model's response,
        stripping optional thinking blocks or Markdown fences.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response.")
        data = json.loads(match.group(0))
        answer = str(data["answer"]).strip().strip("[]").upper()
        explanation = str(data["explanation"]).strip()
        if answer not in self.game.actions:
            raise ValueError(f"Invalid action: {answer}")
        return answer, explanation

    def get_strategy(self) -> tuple[str, str]:
        """
        Extract the final binding strategy after negotiation ends.
        """
        user = "JUDGE"
        actions_list = list(self.game.actions)
        final_message = f"""
The negotiation phase is now closed, and you can no longer communicate with [{self.opponent_id}]. It is time to make your final, binding decision for this one-shot Prisoner's Dilemma.
Respond strictly in the following standardized JSON format and nothing else:
{{"explanation": "<concise explanation of the reasoning behind your chosen answer>", "answer": "<{actions_list[0]} or {actions_list[1]}>"}}
The value for "answer" MUST be either "{actions_list[0]}" or "{actions_list[1]}".
""".strip()
        self.listen(user=user, prompt=final_message)
        raw_response = self.respond(json_mode=True)
        try:
            answer, explanation = self._parse_json_response(raw_response)
        except Exception:
            critical_message = (
                f"Sorry, your message does not follow the required JSON format. "
                f'Please provide your answer strictly as: {{"explanation": "<explanation of the answer>", "answer": "<{actions_list[0]} or {actions_list[1]}>"}}'
            )
            answer, explanation = self._error_handling(critical_message, user=user)
        return answer, explanation

    def _clean_candidate(self, raw_text: str) -> str:
        """Remove harmless model prefixes/thinking blocks before validation."""
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # If the model repeats its own prefix [1234]: at the beginning, remove it
        text = re.sub(rf"^\s*\[?{self.id}\]?\s*:\s*", "", text)
        return text.strip()

    def _record_sanity_fail(self, reason: str) -> bool:
        self.sanity_fail_reasons[reason] = self.sanity_fail_reasons.get(reason, 0) + 1
        if self.debug:
            turn = sum(1 for m in self.messages if m["role"] == "assistant") + 1
            raw_text = getattr(self, "_last_raw_text", "")
            print(f"[DEBUG][Agent {self.id}][Turn {turn}][{reason}]: {raw_text!r}")
            os.makedirs("data", exist_ok=True)
            with open("data/sanity_debug.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "agent_id": self.id,
                    "turn": turn,
                    "reason": reason,
                    "raw_text": raw_text,
                }, ensure_ascii=False) + "\n")
        return False

    def _sanity_check(self, raw_text: str) -> bool:
        """
        Validate a negotiation message before adding it to the history.
        Returns True if valid, False if the model must regenerate.
        """
        self._last_raw_text = raw_text  # Store for debugging if needed
        if not raw_text or not isinstance(raw_text, str):
            return self._record_sanity_fail("empty_or_invalid_type")

        cleaned = self._clean_candidate(raw_text)

        # 1. Leak of stray special tokens / internal model channels
        if any(tok in cleaned for tok in ("<channel|>", "<|", "|>")):
            return self._record_sanity_fail("special_tokens")

        # 2. Check for [DONE] tag: not allowed in the first opening turn
        has_done = bool(re.search(r"\[DONE\]", cleaned, flags=re.IGNORECASE))
        agent_turns_so_far = sum(1 for m in self.messages if m["role"] == "assistant")
        if has_done and agent_turns_so_far == 0:
            return self._record_sanity_fail("early_done")

        # Remove [DONE] to check the actual discursive content
        body = re.sub(r"\[DONE\]", "", cleaned, flags=re.IGNORECASE).strip()

        # 3. Empty or non-informative message (e.g., only punctuation, numbers, or whitespace)
        if not body or not re.search(r"[a-zA-Z]{2,}", body):
            return self._record_sanity_fail("non_informative")

        # 4. Degenerate or excessively long response
        if len(body) > 2500:
            return self._record_sanity_fail("too_long")

        # 5. Spoofing of system roles, referee, or opponent speaker prefix at line start
        spoof_tags = "|".join(["SYSTEM", "JUDGE", "SYSTEM_JUDGE", re.escape(str(self.opponent_id))])
        spoof_prefixes = "|".join(["SYSTEM", "JUDGE", "REFEREE", r"PLAYER\s*[AB]"])
        spoof_pattern = rf"(?im)^\s*(?:\[(?:{spoof_tags})\]|(?:{spoof_prefixes}))\s*:"
        if re.search(spoof_pattern, body):
            return self._record_sanity_fail("role_spoof")

        # 6. Direct declaration of the final move (e.g., "Defect", "Cooperate.", "Cooperation")
        actions_regex = "|".join(re.escape(a) for a in self.game.actions) + "|COOPERATION|DEFECTION"
        if re.fullmatch(rf"(?i)({actions_regex})[\s\.\!\?]*", body):
            return self._record_sanity_fail("direct_action")

        # 7. Reserved syntax for the verdict phase (JSON, markdown blocks, [Action: ...] tags)
        if re.search(
            r"(?i)(\[\s*Action\s*:|^\s*(Action|Move|Decision)\s*:|```|\{\s*['\"](answer|explanation)['\"])",
            body,
        ):
            return self._record_sanity_fail("reserved_syntax")

        # 8. Fake internal thoughts or meta-commentary about the game ending, waiting for a judge, or score reporting
        meta_prefix = (
            r"^\s*[\(\[\*]+\s*(Internal|Self-Correction|Silence|Wait|Waiting|Game|"
            r"No further|Silent|Score|Scores|End of|I remain|The game)"
        )
        if re.match(meta_prefix, body, flags=re.IGNORECASE):
            return self._record_sanity_fail("meta_commentary")
        if re.fullmatch(r"\*?\(.*\)\*?", body, flags=re.DOTALL):
            return self._record_sanity_fail("meta_parenthetical")

        # 9. Prompt leakage: the model should not output phrases that reveal its internal role or the negotiation phase
        leak_snippets = (
            "current phase — pre-game negotiation",
            "do not output json, final move declarations",
            "autonomous agent participating in a behavioral economics",
        )
        if any(s in body.lower() for s in leak_snippets):
            return self._record_sanity_fail("prompt_leakage")

        # 10. Duplicate consecutive message
        prev_assistant_msgs = [m["content"] for m in self.messages if m["role"] == "assistant"]
        if prev_assistant_msgs:
            last_own = re.sub(rf"^\[{self.id}\]:\s*", "", prev_assistant_msgs[-1])
            last_own = re.sub(r"\[DONE\]", "", last_own, flags=re.IGNORECASE)
            if re.sub(r"\W+", "", body.lower()) == re.sub(r"\W+", "", last_own.lower()):
                return self._record_sanity_fail("duplicate_message")
        return True

    def is_done(self) -> bool:
        """
        Check if both the current agent's last message and the opponent's last message contain [DONE].
        """
        own_msgs = [m["content"] for m in self.messages if m["role"] == "assistant"]
        opp_msgs = [
            m["content"] for m in self.messages
            if m["role"] == "user" and m["content"].startswith(f"[{self.opponent_id}]:")
        ]
        if not own_msgs or not opp_msgs:
            return False

        own_done = bool(re.search(r"\[DONE\]", own_msgs[-1], flags=re.IGNORECASE))
        opp_done = bool(re.search(r"\[DONE\]", opp_msgs[-1], flags=re.IGNORECASE))
        return own_done and opp_done

    def _error_handling(self, critical_message: str, user: str) -> tuple[str, str]:
        """
        Handle errors in the agent's JSON response by prompting once more.
        """
        self.listen(user=user, prompt=critical_message)
        raw_response = self.respond(json_mode=True)
        try:
            return self._parse_json_response(raw_response)
        except Exception:
            critical_error_explanation = "The agent failed to provide a valid response in the required format after two attempts."
            return "[ERROR]", critical_error_explanation