import os
import pandas as pd
import uuid
from google.cloud import bigquery
from openai import AzureOpenAI
from google.oauth2 import service_account
import json
from typing import Union, Tuple, List
import logging
from datetime import datetime
from tqdm import tqdm
from utils import TablesAnalyzer
from dotenv import load_dotenv
load_dotenv()

import json

def load_jsonl(file_path):
    """Load a JSON Lines file into a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    

class MultiTableRecursiveSemanticParser:
    def __init__(self, openai_api_key: str = None, temp_dataset: str = "temp_recursive_parser", project_id: str = "cs224v-recursive-parsing"):
        """
        Initialize the recursive semantic parser for multiple tables.
        
        Args:
            openai_api_key: OpenAI API key
            temp_dataset: BigQuery dataset for storing intermediate tables
            project_id: Google Cloud project ID for BigQuery
        """
        self.project_id = project_id
        self.openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", openai_api_key),
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )
        credentials = service_account.Credentials.from_service_account_file('spider-471218-77fec1ca4fcf.json')

        self.bq_client = bigquery.Client(credentials=credentials)
        self.temp_dataset = temp_dataset
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create temp dataset if it doesn't exist
        self._create_temp_dataset()
        
        # Track created tables for cleanup
        self.intermediate_tables = []
    
    def _create_temp_dataset(self):
        """Create the temporary dataset for intermediate tables if it doesn't exist."""
        dataset_id = f"{self.project_id}.{self.temp_dataset}"
        
        try:
            self.bq_client.get_dataset(dataset_id)
            self.logger.info(f"Using existing dataset: {dataset_id}")
        except Exception:
            # Dataset doesn't exist, create it
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset.description = "Temporary dataset for recursive semantic parser intermediate tables"
            
            # Set expiration for tables in this dataset (24 hours)
            dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000
            
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            self.logger.info(f"Created dataset: {dataset_id}")
    
    def query_openai_o3(self, prompt: str, system_prompt: str = "") -> str:
        """
        Query OpenAI o3 model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            Model response as string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.openai_client.chat.completions.create(
                model="o3",
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_query_simple(self, query: str, tables_description: str) -> bool:
        """
        Determine if the query is simple enough to execute directly across multiple tables.
        
        Args:
            query: Natural language query
            tables_description: Description of all tables and their schemas
            
        Returns:
            True if query is simple, False if it needs decomposition
        """
        system_prompt = """You are an expert SQL analyst. Your task is to determine if a natural language query involving multiple tables is simple enough to be converted directly to SQL, or if it needs to be decomposed into simpler sub-queries.

A query is considered SIMPLE if:
- It involves basic SELECT, WHERE, GROUP BY, ORDER BY operations
- It uses simple aggregations (COUNT, SUM, AVG, MAX, MIN)
- It involves straightforward joins (2-3 tables with clear relationships)
- No complex nested subqueries or window functions needed
- No complex business logic or multi-step reasoning
- The joins and relationships are straightforward

A query is considered COMPLEX if:
- It requires multiple aggregation levels
- It needs complex joins across many tables (4+ tables)
- It involves complex business logic or multi-step calculations
- It requires nested subqueries or CTEs with multiple levels
- The logic cannot be expressed in a single, straightforward SQL statement
- It requires temporary intermediate results for further processing
- It involves complex analytical operations across multiple table groups

Return only 'SIMPLE' or 'COMPLEX'."""

        prompt = f"""Query: {query}

Tables Description: {tables_description}

Is this query SIMPLE or COMPLEX?"""

        response = self.query_openai_o3(prompt, system_prompt)
        return response.strip().upper() == 'SIMPLE'
    
    def generate_multi_table_decomposition(self, query: str, tables: List[str], tables_description: str) -> Tuple[str, List[str], str, List[str]]:
        """
        Decompose a complex multi-table query into simpler sub-queries with table assignments.
        
        Args:
            query: Complex natural language query
            tables: List of table names
            tables_description: Description of all tables and their schemas
            
        Returns:
            Tuple of (sub_query_nl, sub_query_tables, final_query_nl, final_query_tables)
        """
        system_prompt = """You are an expert SQL analyst. Your task is to decompose a complex natural language query involving multiple tables into two parts:

1. A SUB_QUERY: A simpler natural language query that will generate an intermediate table
2. A FINAL_QUERY: A natural language query that operates on the intermediate table to get the final answer

The decomposition should:
- Split the tables logically between the sub-query and final query
- The sub-query should handle one major aspect (filtering, initial joins, aggregations, etc.)
- The sub-query should generate a table that contains all necessary data for the final step
- The final query should operate on the intermediate table plus any remaining original tables if needed
- Reduce the complexity by handling the most complex operations in separate steps

Format your response as JSON:
{
    "sub_query": "Natural language description of the sub-query",
    "sub_query_tables": ["list", "of", "table", "names", "for", "sub", "query"],
    "final_query": "Natural language description of the final query on the intermediate table",
    "final_query_tables": ["intermediate_table", "plus", "any", "additional", "original", "tables"]
}

Note: The final_query_tables will automatically include the intermediate table name, so focus on any additional original tables needed."""

        tables_list = "\n".join([f"- {table}" for table in tables])
        
        prompt = f"""Original Query: {query}

Available Tables:
{tables_list}

Tables Description: {tables_description}

Decompose this complex query into a sub-query and final query with appropriate table assignments."""

        response = self.query_openai_o3(prompt, system_prompt)
        
        try:
            decomposition = json.loads(response)
            return (
                decomposition['sub_query'], 
                decomposition['sub_query_tables'],
                decomposition['final_query'], 
                decomposition['final_query_tables']
            )
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse decomposition response: {response}")
            raise ValueError("Failed to parse query decomposition")
    
    def generate_multi_table_sql(self, query: str, tables: List[str], tables_description: str) -> str:
        """
        Generate SQL query from natural language for multiple tables.
        
        Args:
            query: Natural language query
            tables: List of table names
            tables_description: Description of all tables and their schemas
            
        Returns:
            SQL query string (SELECT statement only)
        """
        system_prompt = f"""You are an expert SQL developer. Generate a SQL query based on the natural language request using multiple tables.

CRITICAL REQUIREMENTS:
- Generate ONLY a SELECT statement, never CREATE TABLE or CREATE OR REPLACE TABLE
- The tables are BigQuery tables
- Return only the SELECT SQL query, no explanations or additional statements
- Use standard BigQuery SQL syntax
- Be precise with column names and data types
- Use backticks for table names if they contain special characters
- Properly handle JOINs between tables based on the schema information provided
- Do NOT include any CREATE, INSERT, UPDATE, DELETE, or DDL statements
- Start your response with SELECT (or WITH for CTEs followed by SELECT)
- Infer appropriate join conditions from the table schemas and query context

Available Tables:
{chr(10).join([f'- `{table}`' for table in tables])}

Example format:
SELECT t1.column1, t2.column2, COUNT(*) as count_col
FROM `table1` t1
JOIN `table2` t2 ON t1.id = t2.table1_id
WHERE condition
GROUP BY t1.column1, t2.column2
ORDER BY count_col DESC"""

        prompt = f"""Natural Language Query: {query}

Tables Description: {tables_description}

Available Tables: {', '.join([f'`{table}`' for table in tables])}

Generate ONLY the SELECT SQL query (no CREATE TABLE statements):"""

        sql_response = self.query_openai_o3(prompt, system_prompt)
        
        # Clean and validate the response
        sql_response = sql_response.strip()
        
        # Remove any CREATE TABLE statements if they somehow got included
        if sql_response.upper().startswith('CREATE'):
            # Try to extract just the SELECT part
            lines = sql_response.split('\n')
            select_start = None
            for i, line in enumerate(lines):
                if line.strip().upper().startswith('SELECT') or line.strip().upper().startswith('WITH'):
                    select_start = i
                    break
            
            if select_start is not None:
                sql_response = '\n'.join(lines[select_start:])
                self.logger.warning("Removed CREATE TABLE statement from LLM response, extracted SELECT portion")
            else:
                raise ValueError("LLM generated CREATE TABLE instead of SELECT statement and no SELECT found")
        
        return sql_response
    
    def execute_bigquery(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query on BigQuery and return results as DataFrame.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            query_job = self.bq_client.query(sql_query)
            results = query_job.result()
            return results.to_dataframe()
        except Exception as e:
            self.logger.error(f"BigQuery execution error: {e}")
            self.logger.error(f"Query: {sql_query}")
            raise
    
    def create_intermediate_table(self, sql_query: str, base_table_name: str = None) -> str:
        """
        Execute SQL query and store results in a new BigQuery table.
        
        Args:
            sql_query: SQL query to execute
            base_table_name: Optional base name for the intermediate table
            
        Returns:
            Full name of the created intermediate table
        """
        # Generate unique table name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        if base_table_name:
            table_name = f"{base_table_name}_{timestamp}_{unique_id}"
        else:
            table_name = f"intermediate_{timestamp}_{unique_id}"
        
        full_table_name = f"{self.project_id}.{self.temp_dataset}.{table_name}"
        
        # Create table using CTAS (CREATE TABLE AS SELECT)
        clean_sql_query = sql_query.rstrip(';').strip()
        create_table_sql = f"CREATE TABLE `{full_table_name}` AS {clean_sql_query}"
        
        try:
            self.logger.info(f"Creating intermediate table: {full_table_name}")
            self.logger.info(f"SQL: {create_table_sql}")
            
            query_job = self.bq_client.query(create_table_sql)
            query_job.result()  # Wait for completion
            
            # Track the table for potential cleanup
            self.intermediate_tables.append(full_table_name)
            
            self.logger.info(f"Successfully created intermediate table: {full_table_name}")
            return full_table_name
            
        except Exception as e:
            self.logger.error(f"Error creating intermediate table: {e}")
            self.logger.error(f"SQL: {create_table_sql}")
            raise
    
    def get_tables_description(self, tables: List[str]) -> str:
        """
        Get comprehensive description of multiple tables using TablesAnalyzer.
        
        Args:
            tables: List of table names
            
        Returns:
            Combined description of all tables
        """
        try:
            # Use TablesAnalyzer to get detailed descriptions
            table_analyzer = TablesAnalyzer(tables, self.openai_client.api_key)
            analysis_result = table_analyzer.describe_tables()
            return table_analyzer.return_analysis_summary(analysis_result)
        except Exception as e:
            self.logger.error(f"Error getting tables description: {e}")
            # Fallback to basic schema information
            return self._get_basic_tables_description(tables)
    
    def _get_basic_tables_description(self, tables: List[str]) -> str:
        """
        Fallback method to get basic table descriptions.
        
        Args:
            tables: List of table names
            
        Returns:
            Basic description of all tables
        """
        descriptions = []
        
        for table_name in tables:
            try:
                table_ref = self.bq_client.get_table(table_name)
                
                description = f"\nTable: {table_name}\n"
                description += f"Rows: {table_ref.num_rows:,}\n"
                description += f"Columns: {len(table_ref.schema)}\n"
                description += "Schema:\n"
                
                for field in table_ref.schema:
                    field_type = field.field_type
                    mode = field.mode if field.mode != 'NULLABLE' else ''
                    mode_str = f" ({mode})" if mode else ""
                    description += f"- {field.name}: {field_type}{mode_str}\n"
                
                descriptions.append(description)
                
            except Exception as e:
                self.logger.warning(f"Could not get schema for {table_name}: {e}")
                descriptions.append(f"\nTable: {table_name}\nError: {str(e)}\n")
        
        return "\n".join(descriptions)
    
    def solve_multi_table_query(self, query: str, tables: List[str], tables_description: str = None, 
                               depth: int = 0, output_dir: str = None) -> pd.DataFrame:
        """
        Recursively solve queries involving multiple tables using tail recursion approach.
        
        Args:
            query: Natural language query
            tables: List of BigQuery table names
            tables_description: Description of the tables (will be auto-generated if None)
            depth: Current recursion depth (for logging)
            output_dir: Directory to save intermediate results
            
        Returns:
            Final query results as DataFrame
        """
        self.logger.info(f"{'  ' * depth}Solving multi-table query (depth {depth}): {query[:100]}...")
        self.logger.info(f"{'  ' * depth}Using tables: {', '.join(tables)}")
        
        # Auto-generate table description if not provided
        if tables_description is None:
            self.logger.info(f"{'  ' * depth}Generating description for {len(tables)} tables...")
            tables_description = self.get_tables_description(tables)
        
        # Check if query is simple enough
        if self.is_query_simple(query, tables_description):
            self.logger.info(f"{'  ' * depth}Query is simple, generating multi-table SQL...")
            
            # Generate and execute SQL
            sql_query = self.generate_multi_table_sql(query, tables, tables_description)
            self.logger.info(f"{'  ' * depth}Generated SQL: {sql_query}")
            
            # Execute the query and return results
            result = self.execute_bigquery(sql_query)
            
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Save SQL query and results
                with open(os.path.join(output_dir, f"sql_query.sql"), "w") as f:
                    f.write(sql_query)
                    
                result.to_csv(os.path.join(output_dir, f"execution_results.csv"), index=False)
                
                # Save table information
                with open(os.path.join(output_dir, f"tables_used.json"), "w") as f:
                    json.dump({"tables": tables}, f, indent=2)

            self.logger.info(f"{'  ' * depth}Query executed successfully, returning {len(result)} rows")
            return result
        
        else:
            self.logger.info(f"{'  ' * depth}Query is complex, decomposing...")
            
            # Decompose the query with table assignments
            sub_query, sub_query_tables, final_query, final_query_tables = self.generate_multi_table_decomposition(
                query, tables, tables_description
            )
            
            self.logger.info(f"{'  ' * depth}Sub-query: {sub_query}")
            self.logger.info(f"{'  ' * depth}Sub-query tables: {', '.join(sub_query_tables)}")
            self.logger.info(f"{'  ' * depth}Final query: {final_query}")
            self.logger.info(f"{'  ' * depth}Final query additional tables: {', '.join(final_query_tables)}")
            
            # Generate SQL for the sub-query
            sub_tables_description = self.get_tables_description(sub_query_tables)
            sub_sql = self.generate_multi_table_sql(sub_query, sub_query_tables, sub_tables_description)
            self.logger.info(f"{'  ' * depth}Generated sub-query SQL: {sub_sql}")
            
            # Create intermediate table from sub-query
            intermediate_table = self.create_intermediate_table(
                sub_sql, 
                base_table_name=f"multitable_subquery_depth_{depth}"
            )
            
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save sub-query information
                with open(os.path.join(output_dir, f"sub_query.sql"), "w") as f:
                    f.write(sub_sql)
                    
                # Save decompositions
                decomposition_info = {
                    "sub_query": sub_query,
                    "sub_query_tables": sub_query_tables,
                    "final_query": final_query,
                    "final_query_tables": final_query_tables,
                    "intermediate_table": intermediate_table
                }
                with open(os.path.join(output_dir, f"decompositions.json"), "w") as f:
                    json.dump(decomposition_info, f, indent=2)

                # Save intermediate table results
                df = self.bq_client.query(f"SELECT * FROM `{intermediate_table}`").result().to_dataframe()
                df.to_csv(os.path.join(output_dir, f"intermediate_results.csv"), index=False)

            # Prepare tables for final query (intermediate table + any additional original tables)
            final_tables = [intermediate_table] + [t for t in final_query_tables if t != "intermediate_table"]
            
            # Generate description for the final query tables
            final_tables_description = self.get_tables_description(final_tables)
            
            next_output_dir = os.path.join(output_dir, f"depth_{depth + 1}") if output_dir else None
            
            # Recursively solve the final query
            final_result = self.solve_multi_table_query(
                final_query, 
                final_tables, 
                final_tables_description, 
                depth + 1,
                output_dir=next_output_dir
            )
            
            return final_result
    
    def cleanup_intermediate_tables(self):
        """
        Clean up all intermediate tables created during the recursive process.
        """
        self.logger.info("Cleaning up intermediate tables...")
        
        for table_name in self.intermediate_tables:
            try:
                self.bq_client.delete_table(table_name, not_found_ok=True)
                self.logger.info(f"Deleted intermediate table: {table_name}")
            except Exception as e:
                self.logger.warning(f"Could not delete table {table_name}: {e}")
        
        self.intermediate_tables.clear()
        self.logger.info("Cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_intermediate_tables()


# Example usage
def main():
    # Initialize the parser
    parser = MultiTableRecursiveSemanticParser(
        temp_dataset="temp_recursive_parser",
        project_id="cs224v-recursive-parsing"
    )    
    
    dataset = load_jsonl("spider2-lite.jsonl")
    gold_tables = load_jsonl("gold_tables.jsonl")
    gold_tables = {table['instance_id']: table['gold_tables'] for table in gold_tables}
    
    for d in dataset:
        if d['instance_id'] in gold_tables.keys():
            d['gold_tables'] = gold_tables[d['instance_id']]
            
    dataset = [d for d in dataset if 'gold_tables' in d.keys()]
    # dataset = [d for d in dataset if len(d['gold_tables']) == 1]
    
    gold_path = "./gold"
    
    all_gold_sqls = os.listdir(os.path.join(gold_path, "sql"))
    all_gold_results = os.listdir(os.path.join(gold_path, "exec_result"))

    dataset = [d for d in dataset if f"{d['instance_id']}.sql" in all_gold_sqls]
    dataset = [d for d in dataset if f"{d['instance_id']}.csv" in all_gold_results]
    credentials = service_account.Credentials.from_service_account_file('spider-471218-77fec1ca4fcf.json')
    client = bigquery.Client(credentials=credentials)
    
    def check_for_table_availability(data_point):
        for table_header in data_point['gold_tables']:
            try:
                client.get_table(table_header)
            except Exception as e:
                return False
        return True

    # dataset = [d for d in tqdm(dataset) if check_for_table_availability(d)]
    print("Total number of the questions", len(dataset))
    
    for example in dataset:
        print("Processing example:", example['instance_id'])
        query = example['question']
        tables = example['gold_tables']

        output_path = "./lite_result/o3/" + example['instance_id']
    
        try:
            # Use context manager for automatic cleanup
            with parser:
                result = parser.solve_multi_table_query(
                    query=query, 
                    tables=tables, 
                    output_dir=output_path
                )
                # print("Final Result:")
                # print(result)

            json.dump(example, open(os.path.join(output_path, "metadata.json"), "w"), indent=2)
            break
        
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()