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

    def __init__(self, openai_api_key: str = None):
        """
        Args:
            openai_api_key: Optional OpenAI API key. If None, reads from env variable.
        """
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", openai_api_key),
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )

        # Load domain-specific examples for all 11 domains
        self.domain_examples = {
            "bigquery": self._load_examples("bigquery_examples.json"),
            "dbt": self._load_examples("dbt_examples.json"),
            "airbyte": self._load_examples("airbyte_examples.json"),
            "airflow": self._load_examples("airflow_examples.json"),
            "dagster": self._load_examples("dagster_examples.json"),
            "excel": self._load_examples("excel_examples.json"),
            "jupyter": self._load_examples("jupyter_examples.json"),
            "metabase": self._load_examples("metabase_examples.json"),
            "servicenow": self._load_examples("servicenow_examples.json"),
            "snowflake": self._load_examples("snowflake_examples.json"),
            "superset": self._load_examples("superset_examples.json"),
        }

        # Load all domains metadata
        self.all_domains_config = self._load_examples("all_domains_examples.json")

        # Get list of all supported domains
        all_domains = list(self.all_domains_config.get("domains", {}).keys())

        total_examples = sum(len(ex.get('examples', [])) for ex in self.domain_examples.values())

        print("Initialized Enhanced Multi-Domain VerboseInstructionGenerator")
        print(f"  - Loaded {total_examples} few-shot examples across all 11 domains")
        print(f"  - Supported domains: {', '.join(all_domains)}")
        print(f"  - Schema-aware generation: Enabled (BigQuery)")

    def _load_examples(self, examples_file: str) -> Dict[str, Any]:
        """Load few-shot examples from JSON file."""
        try:
            with open(examples_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Examples file '{examples_file}' not found.")
            return {"examples": [], "common_patterns": {}}

    def _detect_domain(self, task_description: str, tables: List[str]) -> str:
        """
        Detect the domain based on task description and tables using keyword matching.

        Args:
            task_description: Task description text
            tables: List of table names

        Returns:
            Domain name (default: "bigquery")
        """
        task_lower = task_description.lower()

        # Domain-specific keyword mappings (ordered by specificity)
        domain_keywords = {
            "dbt": ["dbt", "snapshot", "seed", "dbt_project", "dbt run", "dbt debug", "jaffle"],
            "airbyte": ["airbyte", "faker", "sync now", "connection"],
            "airflow": ["airflow", "dag", "astro", "airflow ui", "task"],
            "dagster": ["dagster", "asset", "dagster dev", "materialization"],
            "excel": ["excel", "spreadsheet", "pivot table", "formula", "workbook", "cell"],
            "jupyter": ["jupyter", "notebook", "ipynb", ".ipynb", "kernel"],
            "metabase": ["metabase", "question", "metabase ui"],
            "servicenow": ["servicenow", "incident", "service catalog", "cmdb"],
            "snowflake": ["snowflake", "snowsight", "warehouse", "snowflake account"],
            "superset": ["superset", "apache superset", "superset dashboard"],
            "bigquery": ["bigquery", "bq ", "google cloud", "bigquery-public-data"]
        }

        # Check tables for BigQuery-specific patterns
        table_indicators = {
            "bigquery": any("bigquery-public-data" in table.lower() or "." in table for table in tables)
        }

        # Score each domain based on keyword matches
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if domain in table_indicators and table_indicators[domain]:
                score += 2  # Boost score if table patterns match
            if score > 0:
                scores[domain] = score

        # Return domain with highest score
        if scores:
            return max(scores, key=scores.get)

        # Default to bigquery for SQL-related tasks
        return "bigquery"

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
        Generate enhanced verbose step-by-step instructions with multi-domain support.

        Pipeline:
        1. Detect domain (bigquery, dbt, airbyte)
        2. Classify task type
        3. Fetch table schemas (for bigquery only)
        4. Select relevant few-shot examples from domain
        5. Generate instructions with context

        Args:
            task_description: The high-level task to accomplish
            tables: List of table names available (may be empty for non-BigQuery tasks)

        Returns:
            Formatted verbose instruction as string
        """
        # Detect domain
        print("  - Detecting domain...")
        domain = self._detect_domain(task_description, tables)
        print(f"    Domain: {domain}")

        # Classify task type
        print("  - Classifying task type...")
        task_classification = self._classify_task(task_description, tables)
        print(f"    Task type: {task_classification['task_type']}")
        print(f"    Complexity: {task_classification['complexity']}")

        # Get schema information (only for bigquery)
        schema_info = None
        if domain == "bigquery" and tables:
            schema_info = self._get_table_schemas(tables)

        # Build few-shot examples section from appropriate domain
        few_shot_examples = self._build_few_shot_section(domain, task_classification['task_type'])

        # Build enhanced prompt
        system_prompt = self._build_system_prompt(domain, task_classification, few_shot_examples)
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

    def _build_few_shot_section(self, domain: str, task_type: str) -> str:
        """Build few-shot examples section based on domain and task type."""
        domain_examples = self.domain_examples.get(domain, {})
        if not domain_examples.get('examples'):
            return ""

        # Find matching example by task type
        matching_example = None
        for ex in domain_examples['examples']:
            if ex.get('task_type') == task_type:
                matching_example = ex
                break

        # If no exact match, use first example from domain
        if not matching_example and domain_examples['examples']:
            matching_example = domain_examples['examples'][0]

        if not matching_example:
            return ""

        # Format based on domain (BigQuery has tables, others might not)
        tables_line = ""
        if 'tables' in matching_example:
            tables_line = f"\nTables: {', '.join(matching_example.get('tables', []))}"

        return f"""
EXAMPLE REFERENCE:

Task: {matching_example['instruction']}{tables_line}

Verbose Instruction:
{matching_example['verbose_instruction']}

---
Your task should follow a similar detailed, step-by-step format.
"""

    def _build_system_prompt(self, domain: str, task_classification: Dict[str, Any], few_shot_examples: str) -> str:
        """Build enhanced system prompt with domain and task context."""
        task_type = task_classification['task_type']

        # Get domain info from config or use defaults
        domain_config = self.all_domains_config.get("domains", {}).get(domain, {})
        domain_description = domain_config.get("description", f"{domain} tasks")

        # Domain-specific base instructions
        domain_instructions = {
            "bigquery": "You are an expert data analyst creating step-by-step instructions for BigQuery data warehouse and SQL query tasks.",
            "dbt": "You are an expert data engineer creating step-by-step instructions for dbt (data build tool) transformation tasks.",
            "airbyte": "You are an expert data engineer creating step-by-step instructions for Airbyte data integration and pipeline tasks.",
            "airflow": "You are an expert data engineer creating step-by-step instructions for Apache Airflow workflow orchestration tasks.",
            "dagster": "You are an expert data engineer creating step-by-step instructions for Dagster data orchestration tasks.",
            "excel": "You are an expert analyst creating step-by-step instructions for Microsoft Excel spreadsheet operations.",
            "jupyter": "You are an expert data scientist creating step-by-step instructions for Jupyter Notebook data analysis tasks.",
            "metabase": "You are an expert analyst creating step-by-step instructions for Metabase business intelligence and dashboarding tasks.",
            "servicenow": "You are an expert IT professional creating step-by-step instructions for ServiceNow IT service management tasks.",
            "snowflake": "You are an expert data engineer creating step-by-step instructions for Snowflake data warehouse operations.",
            "superset": "You are an expert analyst creating step-by-step instructions for Apache Superset data visualization and dashboarding tasks."
        }

        base_instructions = domain_instructions.get(domain, f"You are an expert creating step-by-step instructions for {domain_description}.")
        base_instructions += "\n\nYour job is to break down a high-level task into detailed, actionable steps that another person could follow, similar to a tutorial or walkthrough."

        # Domain and task-specific guidelines
        domain_guidelines = {
            "bigquery": {
                "web_ui_query_writing": """
For BigQuery Web UI Query Writing tasks:
- Start with navigating to BigQuery console
- Include steps to explore table schemas
- Provide complete SQL query in a code block
- Include steps to execute and save results
- Reference specific UI elements (buttons, panels, menus)""",
                "code_based_querying": """
For BigQuery Code-based Querying tasks:
- Include steps to open/navigate to code editor
- Reference switching between browser and IDE
- Explain schema exploration before coding
- Provide complete, runnable Python code in code blocks
- Include column-level reasoning and business logic
- Add terminal execution steps""",
            },
            "dbt": {
                "project_initialization": """
For dbt Project Initialization tasks:
- Include terminal commands for dbt init
- Reference configuration files (profiles.yml, dbt_project.yml)
- Show vi/editor operations if needed
- Include dbt debug, seed, and run commands
- Mention directory navigation (cd commands)""",
                "source_declaration": """
For dbt Source Declaration tasks:
- Open relevant YAML files in editor
- Show exact YAML configuration to append
- Include dbt seed and dbt run commands
- Reference schema and model structure""",
            },
            "airbyte": {
                "data_pipeline_setup": """
For Airbyte Data Pipeline Setup tasks:
- Include both UI navigation (Airbyte, Snowflake, etc.) and terminal operations
- Reference switching between browser tabs/apps
- Show configuration steps for sources and destinations
- Include environment variable retrieval (echo $VAR)
- Show dbt configuration if part of pipeline
- Include sync/run commands""",
                "data_comparison": """
For Airbyte Data Comparison tasks:
- Include connection configuration exploration
- Show terminal commands for data manipulation
- Reference docker commands if needed
- Include data-diff or comparison tool usage
- Show how to save results to files""",
            }
        }

        # Get guidelines for this domain and task type
        domain_specific = domain_guidelines.get(domain, {})

        # If no specific guidelines, generate generic ones based on domain config
        if not domain_specific or task_type not in domain_specific:
            tools = domain_config.get("tools", [])
            tools_str = ", ".join(tools[:3]) if tools else "relevant tools"
            guidelines = f"""
For {domain} tasks:
- Break down into clear, sequential numbered steps
- Include specific commands and configurations where applicable
- Reference UI elements, buttons, and menu items by name
- Provide complete code/SQL/configuration in markdown code blocks
- Mention tool switching if using multiple tools ({tools_str})
- Include verification steps to confirm success"""
        else:
            guidelines = domain_specific.get(task_type)

        format_rules = """
FORMAT RULES:
- Use numbered steps (1, 2, 3, etc.) or numbered with parentheses (1), 2), 3))
- Start each step with an action verb (Click, Navigate, Write, Execute, Type, Run, etc.)
- Be specific about table/database/file names
- Include exact SQL queries, code, or configuration in markdown code blocks
- Reference specific UI elements (triangles, buttons, panels by name)
- Add reasoning where helpful ("Note that...", "We will use this later...")
- Use collaborative tone occasionally ("First, we check...", "we can see...")
- Start directly with the task overview or step 1 - NO section headers
- For multi-tool tasks, explicitly mention switching between apps ("Switch to", "Go to", "Change to")

DO NOT include:
- Section headers like "TASK:", "THOUGHT:", "STEPS:"
- Bullet points or dashes for main steps (use numbered lists)
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