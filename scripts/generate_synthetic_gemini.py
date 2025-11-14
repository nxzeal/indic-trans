#!/usr/bin/env python3
"""
Generate synthetic casual English translations using Gemini 2.0 Flash.

Optimized for Gemini's rate limits:
- 30 RPM (requests per minute)
- 1,000,000 TPM (tokens per minute)
- 200 RPD (requests per day)

With 50 samples/batch: 10k samples in 200 requests = fits in 200 RPD limit!
Fast generation: ~20 minutes for 10k samples!
"""

import sys
import time
import os
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

# Check for Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Configuration
INPUT_FILE = Path("data/clean/hi_en/splits_research_v3/train/casual_detailed.tsv")
OUTPUT_FILE = Path("data/clean/hi_en/splits_synthetic_casual/train_synthetic.tsv")
BATCH_SIZE = 50  # Optimized for Gemini: 200 batches for 10k samples
MAX_SAMPLES = 10000
MODEL = "models/gemini-2.5-flash"  # Latest fast model (Tier 1)

# Casualization prompt
SYSTEM_PROMPT = """You are an expert at converting formal English to natural, casual spoken English.

Your task: Rewrite formal English sentences into casual, conversational English that sounds like natural speech.

Requirements:
1. CONTRACTIONS: Use contractions aggressively (70%+ coverage)
   - I am → I'm, you are → you're, he is → he's, she is → she's, it is → it's
   - we are → we're, they are → they're, I have → I've, you have → you've
   - will not → won't, do not → don't, does not → doesn't, did not → didn't
   - cannot → can't, could not → couldn't, should not → shouldn't, would not → wouldn't
   - is not → isn't, are not → aren't, was not → wasn't, were not → weren't
   - has not → hasn't, have not → haven't, had not → hadn't

2. COLLOQUIAL PATTERNS:
   - Use "gonna" instead of "going to"
   - Use "wanna" instead of "want to"
   - Use "gotta" instead of "have to" / "have got to"
   - Use "kinda" instead of "kind of"
   - Use "sorta" instead of "sort of"
   - Add filler words naturally: "like", "you know", "I mean", "well", "so"
   - Use "yeah" instead of "yes"

3. CASUAL VOCABULARY: Replace formal words with casual equivalents
   - utilize → use, purchase → buy, obtain → get, assist → help
   - difficult → hard, simple → easy, children → kids, automobile → car
   - residence → home/place, employment → job, commence → start
   - terminate → end/stop, inquire → ask, inform → tell

4. SENTENCE STRUCTURE:
   - Shorter, punchier sentences
   - Start sentences with "So", "Well", "Like"
   - Fragment sentences naturally ("Tired." instead of "I am tired.")
   - Use questions more ("You know what I mean?", "Right?")

5. MAINTAIN MEANING: Keep the core information accurate - casualize style, not content.

Output only the casual versions, one per line, in the same order. No numbering."""


def init_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini client."""
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config,
    )

    return model


def casualize_batch(model: genai.GenerativeModel, formal_sentences: List[str], max_retries: int = 5) -> List[str]:
    """
    Casualize a batch of formal sentences using Gemini with exponential backoff.

    Args:
        model: Gemini model instance
        formal_sentences: List of formal English sentences
        max_retries: Maximum number of retry attempts

    Returns:
        List of casual English sentences (same order)
    """
    # Create numbered list for clarity
    numbered_input = "\n".join(f"{i+1}. {s}" for i, s in enumerate(formal_sentences))

    prompt = f"""{SYSTEM_PROMPT}

Rewrite these formal English sentences into casual, conversational English:

{numbered_input}

Output format: One casual sentence per line, in the same order. No numbering or extra text."""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            casual_text = response.text.strip()
            casual_lines = [line.strip() for line in casual_text.split('\n') if line.strip()]

            # Remove numbering if present
            casual_sentences = []
            for line in casual_lines:
                # Remove leading numbers like "1. ", "2. ", etc.
                if line and line[0].isdigit():
                    line = line.split('. ', 1)[-1] if '. ' in line else line
                casual_sentences.append(line)

            # Validate we got the same number back
            if len(casual_sentences) != len(formal_sentences):
                print(f"Warning: Got {len(casual_sentences)} outputs for {len(formal_sentences)} inputs")
                # Pad or truncate to match
                if len(casual_sentences) < len(formal_sentences):
                    casual_sentences.extend(formal_sentences[len(casual_sentences):])
                else:
                    casual_sentences = casual_sentences[:len(formal_sentences)]

            return casual_sentences

        except Exception as e:
            error_str = str(e)
            # Check for rate limit (429) or quota errors
            if "429" in error_str or "quota" in error_str.lower() or "ResourceExhausted" in error_str:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s, 40s, 80s
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue

            # For non-rate-limit errors or final retry, return original
            print(f"Error during API call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached. Returning original sentences.")
                return formal_sentences


def generate_synthetic_dataset(
    input_file: Path,
    output_file: Path,
    max_samples: int = 10000,
    batch_size: int = 50
) -> None:
    """
    Generate synthetic casual dataset from formal translations.

    Args:
        input_file: Input TSV with columns: hindi, english (formal)
        output_file: Output TSV with columns: hindi, english (casual synthetic)
        max_samples: Maximum number of samples to generate
        batch_size: Number of samples to process per API call
    """
    # Load input data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep='\t', usecols=[0, 1], names=['hindi', 'english'], header=0)

    # Limit to max samples
    if len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")

    print(f"Total samples to generate: {len(df)}")

    # Initialize Gemini
    print("Initializing Gemini...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    model = init_gemini(api_key)
    print(f"Using {MODEL}")

    # Process in batches
    synthetic_data = []
    num_batches = (len(df) + batch_size - 1) // batch_size

    print(f"\nGenerating synthetic casual translations ({batch_size} per batch)...")
    print(f"Rate limiting: ~2 seconds between batches (Tier 1 with exponential backoff)")

    # Prepare output file for incremental saving
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(df), batch_size), total=num_batches):
        batch_df = df.iloc[i:i+batch_size]
        formal_sentences = batch_df['english'].tolist()
        hindi_sentences = batch_df['hindi'].tolist()

        # Casualize the batch with retry logic
        casual_sentences = casualize_batch(model, formal_sentences)

        # Store results in memory
        batch_data = []
        for hindi, casual in zip(hindi_sentences, casual_sentences):
            synthetic_data.append({'hindi': hindi, 'english': casual})
            batch_data.append({'hindi': hindi, 'english': casual})

        # Incremental save: append this batch to file
        batch_df_out = pd.DataFrame(batch_data)
        if i == 0 and not output_file.exists():
            # First batch and file doesn't exist: create new file
            batch_df_out.to_csv(output_file, sep='\t', index=False, header=False, mode='w')
        else:
            # Subsequent batches or file exists: append
            batch_df_out.to_csv(output_file, sep='\t', index=False, header=False, mode='a')

        # Rate limiting: 2 seconds between batches (Tier 1 has higher limits)
        if i + batch_size < len(df):
            time.sleep(2)

    print(f"\n✓ Saved {len(synthetic_data)} synthetic samples to {output_file}")

    # Show sample
    print("\nSample outputs:")
    for i in range(min(5, len(synthetic_data))):
        print(f"\n{i+1}. Hindi: {synthetic_data[i]['hindi']}")
        print(f"   Casual: {synthetic_data[i]['english']}")


def estimate_cost(num_samples: int, batch_size: int = 50) -> None:
    """Estimate generation time and cost."""
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Time estimate with 2s delay between batches
    total_seconds = num_batches * 2
    total_minutes = total_seconds / 60

    # Tier 1 pricing estimate
    # gemini-1.5-flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    # Rough estimate: ~100 tokens input per batch, ~150 tokens output per batch
    avg_input_tokens = 100 * num_batches
    avg_output_tokens = 150 * num_batches
    input_cost = (avg_input_tokens / 1_000_000) * 0.075
    output_cost = (avg_output_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost

    print(f"\n=== Gemini 1.5 Flash Generation (Tier 1) ===")
    print(f"Samples: {num_samples:,}")
    print(f"Batches: {num_batches:,} ({batch_size} samples each)")
    print(f"Model: {MODEL}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Estimated time: {total_minutes:.1f} minutes")
    print(f"Rate limits: Tier 1 (exponential backoff for 429)")
    print(f"===========================================\n")


def main():
    """Main entry point."""
    print("=== Synthetic Casual Dataset Generator (Gemini) ===\n")

    # Check input file exists
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Estimate time
    estimate_cost(MAX_SAMPLES, BATCH_SIZE)

    # Confirm with user
    response = input("Proceed with generation? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)

    # Generate
    generate_synthetic_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_samples=MAX_SAMPLES,
        batch_size=BATCH_SIZE
    )

    print("\n✓ Synthetic dataset generation complete!")
    print(f"\nNext steps:")
    print(f"1. Validate quality: python scripts/verify_research_quality.py --file {OUTPUT_FILE}")
    print(f"2. Create splits and update symlinks")
    print(f"3. Train adapter: python scripts/train_lora.py configs/adapter_casual_detailed.yaml")


if __name__ == "__main__":
    main()
