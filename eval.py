import json
import requests
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
import os
from datetime import datetime

API_URL = "http://localhost:8000/api/ask"
PROGRESS_FILE = "eval_progress.json"
RESULTS_FILE = "eval_results.json"

# Configuration
DELAY_BETWEEN_REQUESTS = 30  # seconds
REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3

def print_header(text):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_progress(current, total, question):
    """Print progress information"""
    percentage = (current / total) * 100
    print(f"\n[{current}/{total}] ({percentage:.1f}%) {question[:60]}...")

def call_api_with_retry(question, max_retries=MAX_RETRIES):
    """Call API with retry logic"""
    for attempt in range(max_retries):
        try:
            res = requests.post(
                API_URL,
                json={
                    "name": "eval",
                    "email": "eval@test.com",
                    "query": question
                },
                timeout=REQUEST_TIMEOUT
            ).json()
            
            # Check for rate limit
            if "error" in res and "rate limit" in str(res.get("error", "")).lower():
                wait_time = 60 * (2 ** attempt)
                print(f"  ⏳ Rate limited! Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            return res, None
            
        except requests.exceptions.ReadTimeout:
            error = f"Request timeout (attempt {attempt + 1}/{max_retries})"
            print(f"  ⚠️ {error}")
            if attempt < max_retries - 1:
                time.sleep(30)
                continue
            return None, error
            
        except Exception as e:
            error = f"Error: {str(e)} (attempt {attempt + 1}/{max_retries})"
            print(f"  ❌ {error}")
            if attempt < max_retries - 1:
                time.sleep(30)
                continue
            return None, error
    
    return None, "Max retries exceeded"

def load_progress():
    """Load existing progress if available"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        print(f"📂 Loaded existing progress: {len(progress['questions'])} questions completed")
        return progress
    return {
        "questions": [],
        "answers": [],
        "contexts": [],
        "ground_truths": [],
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "errors": []
        }
    }

def save_progress(progress):
    """Save progress to file"""
    progress["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def save_final_results(result, progress):
    """Save final evaluation results"""
    final_data = {
        "evaluation_results": result,
        "metadata": {
            "total_questions": len(progress["questions"]),
            "successful_answers": sum(1 for a in progress["answers"] if a),
            "failed_answers": sum(1 for a in progress["answers"] if not a),
            "completion_time": datetime.now().isoformat(),
            "errors": progress["metadata"]["errors"]
        },
        "raw_data": progress
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Final results saved to: {RESULTS_FILE}")

# Main execution
print_header("🚀 RAGAS Evaluation Script")

# Load eval data
print("\n📖 Loading evaluation data...")
with open("eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)
print(f"   Loaded {len(eval_data)} questions")

# Load or initialize progress
progress = load_progress()
start_idx = len(progress["questions"])

if start_idx > 0:
    print(f"   Resuming from question {start_idx + 1}")
else:
    print("   Starting fresh evaluation")

# Process questions
print_header(f"📝 Processing Questions ({start_idx + 1} to {len(eval_data)})")

for i, item in enumerate(eval_data[start_idx:], start_idx + 1):
    print_progress(i, len(eval_data), item["question"])
    
    # Call API
    res, error = call_api_with_retry(item["question"])
    
    # Process response
    if res and "answer" in res and "contexts" in res:
        progress["answers"].append(res["answer"])
        progress["contexts"].append(res["contexts"])
        print("  ✅ Success")
    else:
        progress["answers"].append("")
        progress["contexts"].append([])
        error_msg = f"Question {i}: {error or 'Invalid response'}"
        progress["metadata"]["errors"].append(error_msg)
        print(f"  ⚠️ Failed - using empty response")

    progress["questions"].append(item["question"])
    progress["ground_truths"].append(item["ground_truth"])
    
    # Save progress after each question
    save_progress(progress)
    
    # Delay before next request (except for last question)
    if i < len(eval_data):
        print(f"  ⏸️  Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
        time.sleep(DELAY_BETWEEN_REQUESTS)

# Summary
print_header("📊 Processing Summary")
total = len(progress["questions"])
successful = sum(1 for a in progress["answers"] if a)
failed = total - successful
print(f"  Total Questions: {total}")
print(f"  Successful: {successful} ({(successful/total)*100:.1f}%)")
print(f"  Failed: {failed} ({(failed/total)*100:.1f}%)")

if progress["metadata"]["errors"]:
    print(f"\n  ⚠️  Errors encountered: {len(progress['metadata']['errors'])}")
    for error in progress["metadata"]["errors"][:5]:  # Show first 5
        print(f"     • {error}")
    if len(progress["metadata"]["errors"]) > 5:
        print(f"     ... and {len(progress['metadata']['errors']) - 5} more")

# Verify data integrity
print_header("🔍 Data Integrity Check")
print(f"  Questions: {len(progress['questions'])}")
print(f"  Answers: {len(progress['answers'])}")
print(f"  Contexts: {len(progress['contexts'])}")
print(f"  Ground Truths: {len(progress['ground_truths'])}")

if not all(len(progress[k]) == total for k in ['questions', 'answers', 'contexts', 'ground_truths']):
    print("\n  ❌ ERROR: Data length mismatch!")
    print("  Cannot proceed with evaluation. Please check the data.")
    exit(1)
else:
    print("  ✅ All data lengths match!")

# Create dataset and evaluate
print_header("🎯 Running RAGAS Evaluation")

try:
    dataset = Dataset.from_dict({
        "question": progress["questions"],
        "answer": progress["answers"],
        "contexts": progress["contexts"],
        "ground_truth": progress["ground_truths"]
    })
    
    print("  Evaluating with metrics: faithfulness, answer_relevancy, context_recall")
    print("  This may take a few minutes...\n")
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall
        ]
    )
    
    # Display results
    print_header("✨ EVALUATION RESULTS")
    print(result)
    
    # Save results
    save_final_results(result, progress)
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"🧹 Cleaned up progress file: {PROGRESS_FILE}")
    
    print_header("🎉 Evaluation Complete!")
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    print(f"Progress has been saved to: {PROGRESS_FILE}")
    print("You can resume by running the script again.")
    exit(1)