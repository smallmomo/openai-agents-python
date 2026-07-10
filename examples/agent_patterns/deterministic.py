import asyncio
import json

from pydantic import BaseModel

from agents import Agent, Runner, trace
from examples.auto_mode import input_with_fallback
from examples.volcengine_config import get_model, setup_volcengine

"""
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
"""

story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
    model=None,  # Will be set at runtime after client is configured
)


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool


outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions=(
        "Read the given story outline, and judge the quality. Also, determine if it is a scifi story. "
        "You MUST output ONLY a valid JSON object with the following fields: "
        '{"good_quality": true/false, "is_scifi": true/false}. '
        "Do not include any other text, markdown formatting, or code fences."
    ),
    model=None,  # Will be set at runtime after client is configured
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    model=None,  # Will be set at runtime after client is configured
)


async def main():
    setup_volcengine()
    model = get_model()

    # 为所有 agent 设置火山引擎模型
    story_outline_agent.model = model
    outline_checker_agent.model = model
    story_agent.model = model

    input_prompt = input_with_fallback(
        "What kind of story do you want? ",
        "Write a short sci-fi story.",
    )

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):
        # 1. Generate an outline
        outline_result = await Runner.run(
            story_outline_agent,
            input_prompt,
        )
        print("Outline generated")

        # 2. Check the outline
        outline_checker_result = await Runner.run(
            outline_checker_agent,
            outline_result.final_output,
        )

        # 3. Parse the JSON output (model doesn't support json_schema response_format)
        raw_output = outline_checker_result.final_output.strip()
        # Remove markdown code fences if present
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[-1]
            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]
            raw_output = raw_output.strip()
        checker_output = OutlineCheckerOutput.model_validate_json(raw_output)

        if not checker_output.good_quality:
            print("Outline is not good quality, so we stop here.")
            exit(0)

        if not checker_output.is_scifi:
            print("Outline is not a scifi story, so we stop here.")
            exit(0)

        print("Outline is good quality and a scifi story, so we continue to write the story.")

        # 4. Write the story
        story_result = await Runner.run(
            story_agent,
            outline_result.final_output,
        )
        print(f"Story: {story_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
