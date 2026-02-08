"""Mock LLM for testing CoT faithfulness without API calls."""

from typing import Optional


class MockCoTLLM:
    """Mock LLM that generates CoT responses for testing.

    Supports different reasoning modes:
    - faithful: Reasoning actually drives the answer
    - post_hoc: Answer first, then rationalization
    - shortcut: Uses heuristics, ignores reasoning
    """

    def __init__(self, mode: str = "faithful"):
        """Initialize mock CoT LLM.

        Args:
            mode: Reasoning mode (faithful, post_hoc, shortcut)
        """
        self.mode = mode
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        """Generate mock CoT response.

        Args:
            prompt: Input prompt
            temperature: Temperature (ignored)
            max_tokens: Max tokens (ignored)

        Returns:
            Mock CoT response
        """
        self.call_count += 1

        # Check if this is an intervention (modified reasoning)
        is_intervention = "[MODIFIED]" in prompt or "Actually" in prompt

        # Extract question from prompt
        question = self._extract_question(prompt)

        if is_intervention:
            return self._generate_intervention_response(prompt, question)

        # Generate initial response based on mode
        if self.mode == "faithful":
            return self._generate_faithful_response(question)
        elif self.mode == "post_hoc":
            return self._generate_posthoc_response(question)
        elif self.mode == "shortcut":
            return self._generate_shortcut_response(question)
        else:
            return self._generate_faithful_response(question)

    def _extract_question(self, prompt: str) -> str:
        """Extract question from prompt."""
        # Take first line before "Let's think"
        lines = prompt.split("\n")
        for line in lines:
            if line.strip() and "Let's think" not in line and "Step" not in line:
                return line.strip()
        return "What is 2+2?"

    def _generate_faithful_response(self, question: str) -> str:
        """Generate faithful CoT where reasoning drives answer."""
        # For math questions
        if any(char.isdigit() for char in question):
            return """Step 1: Let me identify the numbers in the question. I see 2 and 2.
Step 2: The operation is addition, indicated by the word 'plus' or symbol '+'.
Step 3: To add 2 + 2, I can count: 2, 3 (that's 2+1), 4 (that's 2+2).
Step 4: Therefore, 2 + 2 = 4.

The answer is 4."""

        # For general questions
        return """Step 1: Let me break down what's being asked.
Step 2: I need to consider the key factors involved.
Step 3: Based on the analysis, the most logical conclusion is...
Step 4: Therefore, this leads to the answer.

The answer is: A well-reasoned conclusion based on the steps above."""

    def _generate_posthoc_response(self, question: str) -> str:
        """Generate post-hoc rationalization (answer first, then reasoning)."""
        # Model "knows" answer first, then rationalizes
        if any(char.isdigit() for char in question):
            return """Step 1: Looking at this problem, the answer is clearly 4.
Step 2: We can verify this by noting that 2 + 2 equals 4.
Step 3: This makes sense because 4 is the sum of 2 and 2.
Step 4: Therefore, as we established at the start, the answer is 4.

The answer is 4."""

        return """Step 1: The obvious answer here is the correct choice.
Step 2: This is evident from the way the question is framed.
Step 3: We can see this is true because it aligns with our initial intuition.
Step 4: Therefore, our initial assessment was correct.

The answer is: The obvious choice."""

    def _generate_shortcut_response(self, question: str) -> str:
        """Generate response where model uses shortcuts, ignoring reasoning."""
        # Model uses pattern matching, not actual reasoning
        if any(char.isdigit() for char in question):
            return """Step 1: This looks like a simple arithmetic problem.
Step 2: For addition problems like this, I'll apply the standard algorithm.
Step 3: However, in this case, I can just recall that 2+2=4.
Step 4: So regardless of the reasoning, the answer is what I know it to be.

The answer is 4."""

        return """Step 1: Let me think about this systematically.
Step 2: Actually, I recognize this type of question.
Step 3: Based on my training, the answer is usually the standard response.
Step 4: So I'll go with that.

The answer is: The standard answer for this question type."""

    def _generate_intervention_response(
        self,
        prompt: str,
        question: str
    ) -> str:
        """Generate response for interventions.

        Args:
            prompt: Modified prompt with intervention
            question: Original question

        Returns:
            Response considering intervention
        """
        # Faithful model: intervention should change answer
        if self.mode == "faithful":
            # If reasoning was modified, answer changes
            if "[MODIFIED]" in prompt:
                return "Given the modified reasoning, the answer would be different: 5"
            else:
                return "4"

        # Post-hoc/shortcut: intervention doesn't change answer (reasoning not used)
        elif self.mode in ["post_hoc", "shortcut"]:
            # Answer stays the same regardless of reasoning modification
            return "4"

        return "4"

    def reset_count(self) -> None:
        """Reset call counter."""
        self.call_count = 0
