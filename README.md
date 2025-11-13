# ReAct Agent for Verbose Instruction Generation

## Objective

Build a **ReAct (Reasoning + Acting) agent** that generates detailed, step-by-step decomposed instructions for BigQuery data analysis tasks, matching the format of `verbose_instruction.txt` files in the [Spider2-V dataset](https://github.com/xlang-ai/Spider2-V).

**Input**: Task description (natural language question)
**Output**: Verbose, tutorial-style instructions (without executing queries)

---

## Approach

### 1. Understanding the Target Format

I first analyzed the target output format from Spider2-V examples:

**Example**: [verbose_instruction.txt from Spider2-V](https://github.com/xlang-ai/Spider2-V/blob/main/evaluation_examples/examples/bigquery/06f5c71a-55b5-4bd7-97bc-1df04fa4463f/verbose_instruction.txt)

**Key Observations**:
- 15-100+ numbered steps
- Starts with UI navigation (console, dataset, tables)
- Includes reasoning steps (identify, verify, determine)
- Breaks down SQL queries into incremental pieces
- Embeds SQL code snippets with explanations
- Uses collaborative language ("First, we...", "Click...", "Write...")
- Very concrete and actionable (no generic placeholders)

### 2. ReAct Agent Design

I implemented a **prompt-based ReAct agent** using Azure OpenAI's "o3" model:

```
┌──────────────────────────────────────────────┐
│  INPUTS                                      │
│  • Task description (question)               │
│  • Available BigQuery tables (gold_tables)   │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  REACT AGENT (Single-Shot Generation)        │
│                                              │
│  System Prompt:                              │
│    "You are an expert data analyst..."       │
│    - Specifies output format                 │
│    - Provides guidelines                     │
│    - Shows example format                    │
│                                              │
│  User Prompt:                                │
│    "Task: [question]"                        │
│    "Available Tables: [tables]"              │
│    "Generate step-by-step instructions"      │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  OUTPUT                                      │
│  Numbered steps (1, 2, 3, ...)               │
│  Saved to: verbose_instruction.txt           │
└──────────────────────────────────────────────┘
```

### 3. Key Design Decisions

#### Decision 1: Single-Shot vs. Multi-Turn ReAct

**Choice**: Single-shot generation

**Reasoning**:
- The "o3" model is powerful enough to generate complete instructions in one pass
- Ground-truth examples are static text, not iterative dialogues
- Simpler, faster, and more consistent output
- Still uses ReAct principles (reasoning about the task, acting by generating steps)

**Alternative Considered**: Multi-turn loop with observation/refinement
**Trade-off**: More complex, slower, but potentially more accurate

---

#### Decision 2: No Query Execution

**Choice**: Generate instructions only (no BigQuery execution)

**Reasoning**:
- Mentor's requirement: "generate verbose instructions from task description"
- Goal is documentation/tutorial, not actual data retrieval
- Matches Spider2-V format (instructions, not results)
- Query execution is handled separately in `main.py` if needed

---

#### Decision 3: Prompt Engineering Strategy

**Choice**: Highly structured system prompt with explicit format guidelines

**Prompt Components**:

1. **Role Definition**: "Expert data analyst creating step-by-step instructions"

2. **Format Specification**:
   - Use numbered steps (1, 2, 3...)
   - Start with action verbs (Click, Navigate, Write, Execute)
   - Be specific about tables and columns
   - Include SQL snippets
   - Use collaborative "we" language

3. **Constraints**:
   - NO section headers like "TASK:", "THOUGHT:", "STEPS:"
   - NO bullet points
   - NO generic placeholders
   - Start directly with step 1

**Reasoning**: Explicit constraints ensure consistent output format matching Spider2-V examples

---

#### Decision 4: Input Context

**Choice**: Provide task description + table names (but not full schemas)

**Reasoning**:
- Table names give the agent enough context to generate relevant instructions
- Full schemas would make prompts too long
- Agent can infer common columns (e.g., `event_name`, `timestamp`) for standard datasets
- Matches how a human expert would approach the task

**Alternative Considered**: Include full table schemas from BigQuery
**Trade-off**: More accurate column references, but longer prompts and higher API costs

---

### 4. Implementation Steps

**Step 1**: Load dataset
- Read task from `spider2-lite.jsonl`
- Get relevant tables from `gold_tables.jsonl`

**Step 2**: Construct prompts
- Build system prompt with format guidelines
- Build user prompt with task + tables

**Step 3**: Generate instructions
- Call Azure OpenAI "o3" model
- Receive numbered steps as plain text

**Step 4**: Save output
- Write to `verbose_instructions/{instance_id}/verbose_instruction.txt`
- Organize by instance ID for easy reference

---

## Example Workflow

### Input
```json
{
  "instance_id": "bq011",
  "question": "How many distinct pseudo users had positive engagement time in the
               7-day period ending on January 7, 2021 at 23:59:59, but had no
               positive engagement time in the 2-day period ending on the same date?",
  "tables": [
    "bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210101",
    "bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210102",
    ... (7 total tables)
  ]
}
```

### Generated Output (Excerpt)
```
1. First, open the BigQuery console and make sure the project is selected.
2. Navigate to the dataset `bigquery-public-data.ga4_obfuscated_sample_ecommerce`...
3. Verify the seven daily partitioned tables are present: events_20210101 through events_20210107.
4. Identify the key columns we will need:
   • event_name – to isolate the user_engagement event
   • event_params – a repeated RECORD containing engagement_time_msec
   • pseudo_user_id – the user identifier we will count
5. Click the "+ Compose new query" button to open a blank query editor.
6. Write a Common Table Expression (CTE) that unions the seven daily tables...
[... 15 total steps with SQL code snippets ...]
```

Full example: [verbose_instructions/bq011/verbose_instruction.txt](verbose_instructions/bq011/verbose_instruction.txt)

---

## Evaluation Criteria

### What Makes a Good Verbose Instruction?

✅ **Completeness**: Covers all steps from start to finish
✅ **Specificity**: Uses actual table/column names, not placeholders
✅ **Clarity**: Each step is actionable and unambiguous
✅ **Structure**: Logical flow from exploration → query building → execution
✅ **SQL Quality**: Correct syntax and logic in embedded code snippets
✅ **Format Consistency**: Matches Spider2-V examples (numbered, no headers)

### Current Performance

Based on initial testing with `bq011`:
- ✅ Generates 15+ detailed steps
- ✅ Includes proper SQL code snippets
- ✅ Follows numbered format without headers
- ✅ Uses specific table/column names
- ✅ Includes UI navigation steps
- ✅ Provides validation steps

### Limitations

⚠️ **No Schema Validation**: Agent doesn't verify column names exist
⚠️ **No SQL Execution**: Generated SQL might have syntax errors
⚠️ **No Comparison to Ground Truth**: Haven't measured similarity to original verbose_instruction.txt files
⚠️ **Single Example Tested**: Only validated on one task so far

---

## Next Steps & Open Questions

### Immediate Next Steps
1. **Batch Processing**: Run on all examples in spider2-lite.jsonl
2. **Quality Check**: Manually compare generated vs. ground-truth instructions
3. **Error Analysis**: Identify common failure patterns

### Questions for Mentor Discussion

**Q1: Format & Granularity**
Is the level of detail appropriate? Should instructions be more/less granular?

**Q2: ReAct Pattern**
Is single-shot generation sufficient, or should I implement a true iterative ReAct loop with:
- Thought: "What should I do next?"
- Action: "Generate next instruction step"
- Observation: "Does this make sense given the task?"

**Q3: Schema Information**
Should I include full table schemas in the prompt for better accuracy, or is table name sufficient?

**Q4: Evaluation Metrics**
How should we measure quality? Options:
- Manual review (time-consuming)
- Similarity metrics (BLEU, ROUGE) vs. ground truth
- SQL validity (can the generated SQL run successfully?)
- Human preference studies

**Q5: Scope of Agent**
Should this agent:
- Only generate instructions (current approach)?
- Also validate/execute the SQL it suggests?
- Iteratively refine instructions based on feedback?

---

## Why This Approach?

### Alignment with ReAct Principles

The ReAct framework emphasizes **reasoning** (thinking about the task) and **acting** (generating outputs). This implementation:

- **Reasons**: System prompt guides the model to think like an expert analyst
- **Acts**: Generates concrete, actionable steps
- **Bridges Thought and Action**: Instructions include both reasoning ("Identify the key columns...") and actions ("Click the button...")

### Simplicity First

Started with the simplest viable approach:
- Single-shot generation
- Minimal context (tables only, not full schemas)
- No execution or validation

**Rationale**: Validate the core concept before adding complexity. If single-shot works well, no need for multi-turn loops.

### Extensibility

This design can easily be extended to:
- Multi-turn ReAct loops (if needed)
- Schema-aware generation (if accuracy requires it)
- Execution validation (if SQL correctness is critical)
- Batch processing (for full dataset evaluation)

---

## Summary

**Goal**: Generate verbose, step-by-step instructions for BigQuery tasks
**Approach**: Single-shot prompt-based ReAct agent using Azure OpenAI "o3"
**Status**: Successfully generates instructions matching Spider2-V format
**Next**: Validate on full dataset and refine based on mentor feedback

This agent provides a foundation for automated instruction generation that can be extended based on evaluation results and requirements.