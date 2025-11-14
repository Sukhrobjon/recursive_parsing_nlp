"""
ReAct Agent to Generate Verbose Instructions for BigQuery Tasks
This agent takes a task description and generates step-by-step decomposed instructions
without executing them, matching the format from Spider2-V verbose_instruction.txt files.
"""

import os
import json
import argparse
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class VerboseInstructionGenerator:
    """
    ReAct-style agent that generates detailed step-by-step instructions
    for completing BigQuery data analysis tasks.
    """

    def __init__(self, openai_api_key: str = None):
        """
        Args:
            openai_api_key: Optional OpenAI API key. If None, reads from env variable.
        """
        # Match main.py setup exactly
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", openai_api_key),
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )

        print("Initialized VerboseInstructionGenerator with Azure OpenAI client")

    def generate_verbose_instruction(self, task_description: str, tables: list) -> str:
        """
        Generate verbose step-by-step instructions using ReAct pattern.

        Args:
            task_description: The high-level task to accomplish
            tables: List of BigQuery table names available

        Returns:
            Formatted verbose instruction as string
        """

        # ReAct Prompt: Think step-by-step about how to accomplish the task
        system_prompt = """You are an expert data analyst creating step-by-step instructions for BigQuery data analysis tasks.

Your job is to break down a high-level task into detailed, actionable steps that another analyst could follow, similar to a tutorial or walkthrough.

Generate instructions in this EXACT format:

[Step-by-step numbered list with clear, actionable instructions]

Example format:
1. First, navigate to BigQuery console and locate the dataset containing the tables.
2. Click on the table `table_name` to view its schema and understand the available columns.
3. Identify the key columns needed: column1, column2, column3.
4. Open the query editor and start writing a SQL query.
5. Write a SELECT statement to retrieve the required columns from the table.
[... continue with more steps ...]

Guidelines:
- Use numbered steps (1, 2, 3, etc.)
- Start each step with an action verb (First, Click, Navigate, Write, Execute, etc.)
- Be specific about which tables to use and which columns to access
- Include logical reasoning steps (identify, determine, check, verify)
- Mention specific UI elements when relevant (console, query editor, buttons)
- Break down complex SQL queries into multiple steps
- Include data validation or verification steps
- Use "we" language occasionally to create collaborative tone (e.g., "First, we check...")
- Be concrete and actionable - someone should be able to follow these steps exactly

DO NOT include:
- Section headers like "TASK:", "THOUGHT:", "STEPS:"
- Bullet points or dashes
- Overly generic instructions
- Just start directly with step 1"""

        tables_str = "\n".join([f"  - {table}" for table in tables])

        user_prompt = f"""Task: {task_description}

Available Tables:
{tables_str}

Please provide detailed step-by-step instructions for completing this task. Start directly with step 1."""

        try:
            response = self.client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )

            verbose_instruction = response.choices[0].message.content.strip()
            return verbose_instruction

        except Exception as e:
            print(f"Error generating verbose instruction: {e}")
            raise

    def save_verbose_instruction(self, verbose_instruction: str, output_path: str):
        """Save the verbose instruction to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(verbose_instruction)

        print(f"Saved verbose instruction to: {output_path}")


def process_datasets(generator: VerboseInstructionGenerator, num_datasets: int = None):
    """
    Process examples from spider2-lite.jsonl.

    Args:
        generator: VerboseInstructionGenerator instance
        num_datasets: Number of datasets to process. If None, process all datasets.
    """
    # Load all examples from the dataset
    try:
        with open('spider2-lite.jsonl', 'r') as f:
            examples = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: spider2-lite.jsonl not found")
        return

    # Load gold tables for all examples
    try:
        with open('gold_tables.jsonl', 'r') as f:
            gold_tables_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: gold_tables.jsonl not found")
        return

    gold_tables_dict = {item['instance_id']: item['gold_tables'] for item in gold_tables_data}

    # Determine how many examples to process
    if num_datasets is None:
        examples_to_process = examples
        print(f"Processing ALL {len(examples)} examples...")
    else:
        examples_to_process = examples[:num_datasets]
        print(f"Processing {len(examples_to_process)} out of {len(examples)} examples...")

    print("=" * 80)

    success_count = 0
    fail_count = 0
    show_preview = (num_datasets == 1)  # Show preview only when processing 1 example

    for idx, example in enumerate(examples_to_process, 1):
        instance_id = example['instance_id']
        question = example['question']
        tables = gold_tables_dict.get(instance_id, [])

        if not tables:
            print(f"[{idx}/{len(examples_to_process)}] Skipping {instance_id}: No gold tables found")
            fail_count += 1
            continue

        print(f"\n[{idx}/{len(examples_to_process)}] Processing: {instance_id}")
        if show_preview:
            print(f"Question: {question}")
        else:
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        print(f"Tables: {len(tables)} tables")

        # Generate verbose instruction
        if show_preview:
            print("Generating verbose instruction...")

        try:
            verbose_instruction = generator.generate_verbose_instruction(question, tables)

            # Save to file
            output_path = f"./verbose_instructions/{instance_id}/verbose_instruction.txt"
            generator.save_verbose_instruction(verbose_instruction, output_path)

            # Show preview for single example
            if show_preview:
                lines = verbose_instruction.split('\n')
                preview_lines = lines[:5]  # Show first 5 lines

                print("\n" + "=" * 80)
                print("GENERATED VERBOSE INSTRUCTION (Preview):")
                print("=" * 80)
                for line in preview_lines:
                    print(line)
                if len(lines) > 5:
                    print(f"... ({len(lines) - 5} more lines)")
                print("=" * 80)
                print(f"\n✓ Complete! Verbose instruction saved to {output_path}")
                print(f"  Total lines: {len(lines)}")
            else:
                print(f"✓ Success! Saved to {output_path}")

            success_count += 1

        except Exception as e:
            print(f"✗ Failed: {e}")
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"COMPLETE!")
    print(f"Successfully processed: {success_count}/{len(examples_to_process)}")
    print(f"Failed: {fail_count}/{len(examples_to_process)}")
    print("=" * 80)


def main():
    """
    Main function with CLI argument parsing.

    Usage:
        python react_verbose_simple.py --num 1      # Run 1 example for testing
        python react_verbose_simple.py --num 10     # Run 10 examples
        python react_verbose_simple.py --all        # Run all datasets
    """
    parser = argparse.ArgumentParser(
        description="Generate verbose instructions for BigQuery tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a single example (shows preview)
  python react_verbose_simple.py --num 1

  # Process 10 examples
  python react_verbose_simple.py --num 10

  # Process first 100 examples
  python react_verbose_simple.py --num 100

  # Process all datasets
  python react_verbose_simple.py --all
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--num',
        type=int,
        metavar='N',
        help='Number of datasets to process (e.g., 1, 10, 100)'
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets'
    )

    args = parser.parse_args()

    # Initialize generator (same setup as main.py)
    generator = VerboseInstructionGenerator()

    if args.all:
        print("Running in ALL mode...")
        process_datasets(generator, num_datasets=None)
    else:
        print(f"Running with {args.num} dataset(s)...")
        process_datasets(generator, num_datasets=args.num)


if __name__ == "__main__":
    main()