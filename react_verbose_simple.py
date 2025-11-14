"""
Enhanced ReAct Agent to Generate Verbose Instructions for BigQuery Tasks
This agent takes a task description and generates step-by-step decomposed instructions
without executing them, matching the format from Spider2-V verbose_instruction.txt files.

Features:
- Task classification using LLM
- Schema-aware instruction generation (integrates with TablesAnalyzer)
- Few-shot learning from curated BigQuery examples
- Multi-pattern support (Web UI, Code-based, Schema modification)
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
from utils import TablesAnalyzer

load_dotenv()


class VerboseInstructionGenerator:
    """
    Enhanced ReAct-style agent that generates detailed step-by-step instructions
    for completing BigQuery data analysis tasks.

    Features:
    - Task classification (Web UI, Code-based, Schema modification)
    - Schema-aware generation using TablesAnalyzer
    - Few-shot learning from example patterns
    """

    def __init__(self, openai_api_key: str = None, examples_file: str = "bigquery_examples.json"):
        """
        Args:
            openai_api_key: Optional OpenAI API key. If None, reads from env variable.
            examples_file: Path to JSON file with few-shot examples.
        """
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", openai_api_key),
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )

        # Load few-shot examples
        self.examples = self._load_examples(examples_file)

        print("Initialized Enhanced VerboseInstructionGenerator")
        print(f"  - Loaded {len(self.examples.get('examples', []))} few-shot examples")
        print(f"  - Schema-aware generation: Enabled")

    def _load_examples(self, examples_file: str) -> Dict[str, Any]:
        """Load few-shot examples from JSON file."""
        try:
            with open(examples_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Examples file '{examples_file}' not found. Using basic generation.")
            return {"examples": [], "common_patterns": {}}

    def _classify_task(self, task_description: str, tables: List[str]) -> Dict[str, Any]:
        """
        Classify the task type using LLM.

        Args:
            task_description: The task to classify
            tables: Available tables

        Returns:
            Dictionary with task_type, complexity, and key_operations
        """
        classification_prompt = f"""Analyze this BigQuery task and classify it.

Task: {task_description}

Tables: {', '.join(tables)}

Return a JSON object with:
{{
  "task_type": "web_ui_query_writing" | "code_based_querying" | "schema_modification" | "data_analysis",
  "complexity": "simple" | "medium" | "complex",
  "key_operations": ["operation1", "operation2", ...],
  "requires_code": true/false
}}

Focus on:
- web_ui_query_writing: Writing SQL queries in BigQuery console/UI
- code_based_querying: Writing Python/code to query BigQuery
- schema_modification: Modifying table schemas (adding columns, etc.)
- data_analysis: Complex analytical tasks requiring multiple steps

Key operations might include: "aggregation", "join", "filter", "group_by", "calculation", "time_series", etc.

Return ONLY the JSON object, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model="o3",
                messages=[{"role": "user", "content": classification_prompt}],
            )

            result = response.choices[0].message.content.strip()
            # Extract JSON if wrapped in markdown
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            return json.loads(result)
        except Exception as e:
            print(f"Warning: Task classification failed: {e}")
            # Return default classification
            return {
                "task_type": "web_ui_query_writing",
                "complexity": "medium",
                "key_operations": [],
                "requires_code": False
            }

    def _get_table_schemas(self, tables: List[str]) -> Optional[str]:
        """
        Get schema information for tables using TablesAnalyzer.

        Args:
            tables: List of table names

        Returns:
            Formatted schema information string, or None if failed
        """
        try:
            print("  - Fetching table schemas...")
            analyzer = TablesAnalyzer(tables)
            schema_descriptions = analyzer.describe_tables()

            # Format schema information
            schema_text = []
            for schema_sig, info in schema_descriptions.items():
                schema_text.append(f"\nTables: {', '.join(info['tables'])}")
                schema_text.append(info['formatted_schema'])
                if 'description' in info:
                    schema_text.append(f"\nDescription:\n{info['description']}")

            return "\n".join(schema_text)
        except Exception as e:
            print(f"  - Warning: Could not fetch schemas: {e}")
            return None

    def generate_verbose_instruction(self, task_description: str, tables: list) -> str:
        """
        Generate enhanced verbose step-by-step instructions.

        Pipeline:
        1. Classify task type
        2. Fetch table schemas
        3. Select relevant few-shot examples
        4. Generate instructions with context

        Args:
            task_description: The high-level task to accomplish
            tables: List of BigQuery table names available

        Returns:
            Formatted verbose instruction as string
        """
        print("  - Classifying task type...")
        task_classification = self._classify_task(task_description, tables)
        print(f"    Task type: {task_classification['task_type']}")
        print(f"    Complexity: {task_classification['complexity']}")

        # Get schema information
        schema_info = self._get_table_schemas(tables)

        # Build few-shot examples section
        few_shot_examples = self._build_few_shot_section(task_classification['task_type'])

        # Build enhanced prompt
        system_prompt = self._build_system_prompt(task_classification, few_shot_examples)
        user_prompt = self._build_user_prompt(task_description, tables, schema_info, task_classification)

        print("  - Generating verbose instructions...")
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

    def _build_few_shot_section(self, task_type: str) -> str:
        """Build few-shot examples section based on task type."""
        if not self.examples.get('examples'):
            return ""

        # Find matching example
        matching_example = None
        for ex in self.examples['examples']:
            if ex['task_type'] == task_type:
                matching_example = ex
                break

        # If no exact match, use first example
        if not matching_example and self.examples['examples']:
            matching_example = self.examples['examples'][0]

        if not matching_example:
            return ""

        return f"""
EXAMPLE REFERENCE:

Task: {matching_example['instruction']}

Tables: {', '.join(matching_example['tables'])}

Verbose Instruction:
{matching_example['verbose_instruction']}

---
Your task should follow a similar detailed, step-by-step format.
"""

    def _build_system_prompt(self, task_classification: Dict[str, Any], few_shot_examples: str) -> str:
        """Build enhanced system prompt with task context."""
        task_type = task_classification['task_type']

        # Base instructions
        base_instructions = """You are an expert data analyst creating step-by-step instructions for BigQuery tasks.

Your job is to break down a high-level task into detailed, actionable steps that another analyst could follow, similar to a tutorial or walkthrough."""

        # Task-specific guidelines
        task_guidelines = {
            "web_ui_query_writing": """
For Web UI Query Writing tasks:
- Start with navigating to BigQuery console
- Include steps to explore table schemas
- Provide complete SQL query in a code block
- Include steps to execute and save results
- Reference specific UI elements (buttons, panels, menus)""",
            "code_based_querying": """
For Code-based Querying tasks:
- Include steps to open/navigate to code editor
- Reference switching between browser and IDE
- Explain schema exploration before coding
- Provide complete, runnable code in code blocks
- Include column-level reasoning and business logic
- Add terminal execution steps""",
            "schema_modification": """
For Schema Modification tasks:
- Navigate to specific table in project
- Reference exact UI elements for schema editing
- Be precise about field names and types
- Include save/commit steps""",
            "data_analysis": """
For Data Analysis tasks:
- Break down into logical analytical steps
- Include data exploration phases
- Show intermediate verification steps
- Explain reasoning behind each step"""
        }

        guidelines = task_guidelines.get(task_type, task_guidelines["web_ui_query_writing"])

        format_rules = """
FORMAT RULES:
- Use numbered steps (1, 2, 3, etc.)
- Start each step with an action verb (Click, Navigate, Write, Execute, etc.)
- Be specific about table names and column names
- Include exact SQL queries or code in markdown code blocks
- Reference specific UI elements (triangles, buttons, panels by name)
- Add reasoning where helpful ("Note that...", "We are interested in...")
- Use collaborative tone occasionally ("First, we check...")
- Start directly with the task overview or step 1 - NO section headers

DO NOT include:
- Section headers like "TASK:", "THOUGHT:", "STEPS:"
- Bullet points or dashes (use numbered lists)
- Generic placeholders without specific guidance
"""

        return f"""{base_instructions}

{guidelines}

{format_rules}

{few_shot_examples}"""

    def _build_user_prompt(self, task_description: str, tables: List[str],
                          schema_info: Optional[str], task_classification: Dict[str, Any]) -> str:
        """Build user prompt with all context."""
        tables_str = "\n".join([f"  - {table}" for table in tables])

        prompt_parts = [f"Task: {task_description}\n"]
        prompt_parts.append(f"Task Type: {task_classification['task_type']}")
        prompt_parts.append(f"Complexity: {task_classification['complexity']}")

        if task_classification.get('key_operations'):
            prompt_parts.append(f"Key Operations: {', '.join(task_classification['key_operations'])}")

        prompt_parts.append(f"\nAvailable Tables:\n{tables_str}")

        if schema_info:
            prompt_parts.append(f"\nTable Schema Information:\n{schema_info}")

        prompt_parts.append("\nGenerate detailed step-by-step verbose instructions for this task.")
        prompt_parts.append("Follow the format from the example reference provided.")
        prompt_parts.append("Start directly with the task overview or step 1.")

        return "\n".join(prompt_parts)

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