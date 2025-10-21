from google.cloud import bigquery
from openai import AzureOpenAI
from typing import List, Dict, Tuple, Any, Optional
import hashlib
import json
import logging
from collections import defaultdict
from google.oauth2 import service_account

import pandas as pd
from google.cloud.bigquery import SchemaField
import os


class TablesAnalyzer:
    def __init__(self, table_names: List[str], api_key: str = None):
        """
        Initialize the TablesAnalyzer.
        
        Args:
            client: Pre-initiated BigQuery Client
            table_names: List of table names that can be loaded using client.get_table
            api_key: API key for Azure OpenAI
        """
        credentials = service_account.Credentials.from_service_account_file('spider-471218-77fec1ca4fcf.json')
        self.client = bigquery.Client(credentials=credentials)
        
        self.table_names = table_names
        self.azure_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", api_key),
            api_version="2024-12-01-preview",
            azure_endpoint="https://ovalnairr.openai.azure.com/",
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def _get_schema_signature(self, table_ref: bigquery.Table) -> str:
        """
        Generate a unique signature for a table schema.
        
        Args:
            table_ref: BigQuery table reference
            
        Returns:
            SHA256 hash of the schema as a string
        """
        # Create a consistent representation of the schema
        schema_items = []
        for field in sorted(table_ref.schema, key=lambda x: x.name):
            schema_items.append(f"{field.name}:{field.field_type}:{field.mode}")
        
        schema_string = "|".join(schema_items)
        return hashlib.sha256(schema_string.encode()).hexdigest()
    
    def _get_sample_row(self, table_name: str) -> Dict[str, Any]:
        """
        Get the first row from a table as sample data.
        
        Args:
            table_name: Full table name
            
        Returns:
            Dictionary containing the first row data
        """
        try:
            query = f"""
            SELECT *
            FROM `{table_name}`
            LIMIT 1
            """
            
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Convert to list and get first row
            rows = list(results)
            if rows:
                # Convert Row to dictionary
                return dict(rows[0])
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"Could not fetch sample row for {table_name}: {e}")
            return {}
    
    def _format_schema_with_sample(self, table_ref: bigquery.Table, sample_row: Dict[str, Any]) -> str:
        """
        Format schema information with sample values into a string.
        
        Args:
            table_ref: BigQuery table reference
            sample_row: Sample row data
            
        Returns:
            Formatted string with column info and sample values
        """
        lines = []
        lines.append(f"Table Schema ({len(table_ref.schema)} columns):")
        lines.append("=" * 50)
        
        for field in table_ref.schema:
            column_name = field.name
            column_type = field.field_type
            mode = field.mode if field.mode != 'NULLABLE' else ''
            mode_str = f" ({mode})" if mode else ""
            
            # Get sample value
            sample_value = sample_row.get(column_name, "N/A")
            
            # Format sample value for display
            if sample_value is not None:
                if isinstance(sample_value, str):
                    # Truncate long strings
                    if len(str(sample_value)) > 50:
                        sample_value = str(sample_value)[:47] + "..."
                    sample_value = f'"{sample_value}"'
                else:
                    sample_value = str(sample_value)
            else:
                sample_value = "NULL"
            
            lines.append(f"• {column_name} / {column_type}{mode_str} / Example: {sample_value}")
        
        return "\n".join(lines)
    
    def _query_azure_openai(self, prompt: str, system_prompt: str = "") -> str:
        """
        Query Azure OpenAI for schema description.
        
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
            response = self.azure_client.chat.completions.create(
                model="o3",  # your Azure deployment name
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Azure OpenAI API error: {e}")
            raise
    
    def _generate_schema_description(self, schema_info: str, table_names: List[str]) -> str:
        """
        Generate description for a schema using LLM.
        
        Args:
            schema_info: Formatted schema information with sample values
            table_names: List of table names with this schema
            
        Returns:
            LLM-generated description of the schema
        """
        system_prompt = """You are a data analyst expert. Your task is to analyze database table schemas and provide clear, comprehensive descriptions of what each column represents and the overall purpose of the tables.

For each column, explain:
- What type of data it contains
- What business purpose it serves
- Any relationships or patterns you can infer from the column names and sample values
- Data quality observations (if any)

Provide a professional, concise description that would help someone understand the table structure and its business context."""

        prompt = f"""Please analyze this database schema and provide a comprehensive description:

Tables with this schema: {', '.join(table_names)}

{schema_info}

Please provide only the following:
1. A brief overview of what these tables likely represent
2. Detailed description of each column and its purpose

Format your response in a clear, structured manner."""

        return self._query_azure_openai(prompt, system_prompt)
    
    def describe_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        Group tables by schema and generate descriptions for each group.
        
        Returns:
            Dictionary with schema signatures as keys and description info as values
        """
        self.logger.info(f"Analyzing {len(self.table_names)} tables...")
        
        # Group tables by schema
        schema_groups = defaultdict(list)
        schema_details = {}
        
        for table_name in self.table_names:
            try:
                self.logger.info(f"Processing table: {table_name}")
                
                # Get table reference
                table_ref = self.client.get_table(table_name)
                
                # Generate schema signature
                schema_sig = self._get_schema_signature(table_ref)
                
                # Add to group
                schema_groups[schema_sig].append(table_name)
                
                # Store schema details (use first table in group for sample)
                if schema_sig not in schema_details:
                    # Get sample row
                    sample_row = self._get_sample_row(table_name)
                    
                    # Format schema with sample
                    schema_info = self._format_schema_with_sample(table_ref, sample_row)
                    
                    schema_details[schema_sig] = {
                        'table_ref': table_ref,
                        'schema_info': schema_info,
                        'sample_row': sample_row
                    }
                    
            except Exception as e:
                self.logger.error(f"Error processing table {table_name}: {e}")
                continue
        
        self.logger.info(f"Found {len(schema_groups)} unique schema groups")
        
        # Generate descriptions for each schema group
        results = {}
        
        for schema_sig, table_list in schema_groups.items():
            try:
                self.logger.info(f"Generating description for schema group with {len(table_list)} tables")
                
                schema_detail = schema_details[schema_sig]
                
                # Generate LLM description
                description = self._generate_schema_description(
                    schema_detail['schema_info'], 
                    table_list
                )
                
                results[schema_sig] = {
                    'tables': table_list,
                    'schema_info': schema_detail['schema_info'],
                    'sample_row': schema_detail['sample_row'],
                    'description': description,
                    'table_count': len(table_list)
                }
                
            except Exception as e:
                self.logger.error(f"Error generating description for schema {schema_sig}: {e}")
                results[schema_sig] = {
                    'tables': table_list,
                    'schema_info': schema_detail['schema_info'],
                    'sample_row': schema_detail['sample_row'],
                    'description': f"Error generating description: {str(e)}",
                    'table_count': len(table_list)
                }
        
        return results
        
    
    def print_analysis_summary(self, results: Dict[str, Dict[str, Any]]):
        """
        Print a formatted summary of the analysis results.
        
        Args:
            results: Results from describe_tables()
        """
        print("=" * 80)
        print("TABLES ANALYSIS SUMMARY")
        print("=" * 80)
        
        for i, (schema_sig, info) in enumerate(results.items(), 1):
            print(f"\n📊 SCHEMA GROUP {i}")
            print("-" * 40)
            print(f"Tables ({info['table_count']}):")
            for table in info['tables']:
                print(f"  • {table}")
            
            print(f"\n📝 DESCRIPTION:")
            print(info['description'])
            
            print(f"\n🏗️ SCHEMA DETAILS:")
            print(info['schema_info'])
            
            if i < len(results):
                print("\n" + "="*80)
    
    def export_to_json(self, results: Dict[str, Dict[str, Any]], filename: str):
        """
        Export analysis results to JSON file.
        
        Args:
            results: Results from describe_tables()
            filename: Output JSON filename
        """
        # Convert results to JSON-serializable format
        export_data = {}
        
        for schema_sig, info in results.items():
            export_data[schema_sig] = {
                'tables': info['tables'],
                'description': info['description'],
                'table_count': info['table_count'],
                'schema_info': info['schema_info'],
                # Sample row might contain non-serializable types, convert to strings
                'sample_row': {k: str(v) for k, v in info['sample_row'].items()}
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis results exported to: {filename}")
        
    def return_analysis_summary(self, results: Dict[str, Dict[str, Any]]):
        description = ""

        for i, (schema_sig, info) in enumerate(results.items(), 1):
            description += f"\n📊 SCHEMA GROUP {i}"
            description += "\n" 
            description += f"Tables ({info['table_count']}):"
            for table in info['tables']:
                description += f"  • {table}"
            
            description += f"\n📝 DESCRIPTION:"
            description += info['description']
            
            description += f"\n🏗️ SCHEMA DETAILS:"
            description += info['schema_info']

        return description
    
    
class PandasToBigQuery:
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: Your Google Cloud Project ID
            credentials_path: Path to service account JSON file (optional if using default credentials)
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = bigquery.Client(credentials=credentials)
        
        self.project_id = project_id
    
    def load_csv_to_bigquery(self, csv_file_path: str, dataset_id: str, table_id: str, 
                            if_exists: str = 'replace', auto_detect_schema: bool = True) -> None:
        """
        Load a local CSV file directly to BigQuery.
        
        Args:
            csv_file_path: Path to the local CSV file
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            if_exists: 'replace', 'append', or 'fail'
            auto_detect_schema: Whether to auto-detect schema from CSV
        """
        # Read CSV into DataFrame first
        df = pd.read_csv(csv_file_path)
        
        # Use the DataFrame method
        self.load_dataframe_to_bigquery(df, dataset_id, table_id, if_exists, auto_detect_schema)
    
    def load_dataframe_to_bigquery(self, df: pd.DataFrame, dataset_id: str, table_id: str,
                                  if_exists: str = 'replace', auto_detect_schema: bool = True,
                                  custom_schema: Optional[List[SchemaField]] = None) -> None:
        """
        Load pandas DataFrame to BigQuery table using pandas-gbq.
        
        Args:
            df: Pandas DataFrame to upload
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            if_exists: 'replace', 'append', or 'fail'
            auto_detect_schema: Whether to auto-detect schema
            custom_schema: Custom schema (optional)
        """
        try:
            import pandas_gbq
        except ImportError:
            raise ImportError("Please install pandas-gbq: pip install pandas-gbq")
        
        # Create full table ID
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        print(f"Loading DataFrame to {table_ref}...")
        print(f"DataFrame shape: {df.shape}")
        
        # Configure load job
        pandas_gbq.to_gbq(
            dataframe=df,
            destination_table=f"{dataset_id}.{table_id}",
            project_id=self.project_id,
            if_exists=if_exists,
            table_schema=custom_schema,
            progress_bar=True
        )
        
        print(f"Successfully loaded data to {table_ref}")
    
    def load_with_job_config(self, df: pd.DataFrame, dataset_id: str, table_id: str,
                           custom_schema: Optional[List[SchemaField]] = None,
                           write_disposition: str = 'WRITE_TRUNCATE') -> None:
        """
        Load DataFrame using BigQuery client with job configuration.
        
        Args:
            df: Pandas DataFrame to upload
            dataset_id: BigQuery dataset ID  
            table_id: BigQuery table ID
            custom_schema: Custom schema (optional)
            write_disposition: 'WRITE_TRUNCATE', 'WRITE_APPEND', or 'WRITE_EMPTY'
        """
        # Get table reference
        table_ref = self.client.dataset(dataset_id).table(table_id)
        
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            source_format=bigquery.SourceFormat.PARQUET,  # More efficient than CSV
        )
        
        if custom_schema:
            job_config.schema = custom_schema
        else:
            job_config.autodetect = True
        
        print(f"Loading DataFrame to {self.project_id}.{dataset_id}.{table_id}...")
        
        # Load DataFrame
        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        
        # Wait for job to complete
        job.result()
        
        # Get updated table info
        table = self.client.get_table(table_ref)
        print(f"Loaded {table.num_rows} rows and {len(table.schema)} columns")
    
    def create_dataset_if_not_exists(self, dataset_id: str, location: str = 'US') -> None:
        """
        Create dataset if it doesn't exist.
        
        Args:
            dataset_id: BigQuery dataset ID
            location: Dataset location (default: 'US')
        """
        dataset_ref = self.client.dataset(dataset_id)
        
        try:
            self.client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_id} already exists")
        except Exception:
            # Dataset doesn't exist, create it
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            dataset = self.client.create_dataset(dataset)
            print(f"Created dataset {dataset_id}")
    
    def define_custom_schema(self, df: pd.DataFrame) -> List[SchemaField]:
        """
        Create a custom schema based on DataFrame dtypes.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of BigQuery SchemaField objects
        """
        schema = []
        
        for column, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                bq_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                bq_type = "FLOAT"
            elif pd.api.types.is_bool_dtype(dtype):
                bq_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                bq_type = "TIMESTAMP"
            else:
                bq_type = "STRING"
            
            schema.append(SchemaField(column, bq_type))
        
        return schema
    
    def load_large_dataframe_in_chunks(self, df: pd.DataFrame, dataset_id: str, 
                                     table_id: str, chunk_size: int = 10000) -> None:
        """
        Load large DataFrame in chunks to avoid memory issues.
        
        Args:
            df: Large pandas DataFrame
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            chunk_size: Number of rows per chunk
        """
        total_rows = len(df)
        chunks = [df[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        
        print(f"Loading {total_rows} rows in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            print(f"Loading chunk {i+1}/{len(chunks)} ({len(chunk)} rows)...")
            
            # First chunk replaces, others append
            if_exists = 'replace' if i == 0 else 'append'
            write_disposition = 'WRITE_TRUNCATE' if i == 0 else 'WRITE_APPEND'
            
            self.load_dataframe_to_bigquery(
                chunk, dataset_id, table_id, if_exists=if_exists
            )
        
        print("All chunks loaded successfully!")


# Example usage
def main():
    # Initialize BigQuery client    
    # List of table names to analyze
    table_names = [
        "your-project.dataset1.table1",
        "your-project.dataset1.table2", 
        "your-project.dataset2.table3",
        # Add more table names...
    ]
    
    # Initialize analyzer
    analyzer = TablesAnalyzer(
        table_names=table_names,
        api_key="your-azure-openai-api-key"
    )
    
    # Analyze tables
    try:
        results = analyzer.describe_tables()
        
        # Print summary
        analyzer.print_analysis_summary(results)
        
        # Export to JSON
        analyzer.export_to_json(results, "table_analysis_results.json")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        

if __name__ == "__main__":
    main()