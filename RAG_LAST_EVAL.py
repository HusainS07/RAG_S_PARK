# test_rag.py - Comprehensive RAG System Evaluation
import json
import os
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import httpx

load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if not OPENROUTER_API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env!")
    exit(1)

print("🔧 Setting up RAG evaluation system...\n")

# Initialize LLM for evaluation
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=150,
    request_timeout=30
)

# Test Questions
TEST_QUESTIONS = [
    "How can I cancel an existing parking booking?",
    "Where can I find my active parking bookings in the app?",
    "What details are shown for an active parking booking?",
    "Will I get confirmation after canceling a booking?",
    "How is a refund handled after cancellation?",
    "What should I do if I face issues while canceling a booking?",
    "How do I make a new parking reservation?",
    "Can I filter parking spots while searching?",
    "What should I check before confirming a parking booking?",
    "What do I receive after a successful parking booking?",
    "Why is the booking confirmation important?",
    "Can I extend my parking duration after booking?",
    "What happens if the parking extension option is unavailable?",
    "Why might parking extensions not be allowed?",
    "How can I locate my parked vehicle using the app?",
    "What should I do after parking to help locate my car later?",
    "What if the app does not have a Find My Car feature?",
    "What should I do if a parking payment fails?",
    "How can I resolve a double payment issue?",
    "How long do refunds usually take to process?",
    "How can I avoid payment issues in the future?",
    "What should I do if I receive a parking violation notice?",
    "What evidence is needed to dispute a parking violation?",
    "Who helps resolve parking violations caused by system errors?",
    "What if the parking violation was due to parking in the wrong spot?",
    "Where can I see my payment transaction history?",
    "Is confirmation required before canceling a booking?",
    "What information is required to contact customer support?",
    "What role does a QR code play in parking systems?",
    "Can booking details be modified after confirmation?"
]

print(f"📝 Loaded {len(TEST_QUESTIONS)} test questions\n")

# Helper Functions
def extract_score(response_text):
    """Extract numerical score from LLM response"""
    try:
        return min(1.0, max(0.0, float(response_text.strip())))
    except:
        pass
    
    match = re.search(r'(?:score|rating)?[:\s]*([0-1]\.?\d*)', response_text.lower())
    if match:
        try:
            return min(1.0, max(0.0, float(match.group(1))))
        except:
            pass
    
    return 0.5

def query_rag_system(question):
    """Query the RAG backend API"""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/ask",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "query": question
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "answer": data.get("answer", ""),
                "contexts": data.get("contexts", []),
                "matched": data.get("matched", False)
            }
        else:
            print(f"   ⚠️ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ❌ Connection Error: {str(e)[:50]}")
        return None

def evaluate_faithfulness(question, answer, contexts):
    """
    Evaluate if answer is faithful to the retrieved contexts.
    Score: 0.0 - 1.0
    """
    if not contexts or all(not c.strip() for c in contexts):
        return 0.2
    
    context_str = "\n\n".join(contexts)
    
    prompt = f"""Evaluate how faithful this answer is to the provided context.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER: {answer}

Rate on a scale of 0-1:
- 1.0 = Answer is completely faithful, uses only information from context
- 0.75 = Answer is mostly faithful with minor reasonable inferences
- 0.5 = Answer mixes context with some outside knowledge
- 0.25 = Answer mostly ignores context or adds unrelated info
- 0.0 = Answer contradicts context or fabricates information

Return ONLY the numerical score (e.g., 0.85)"""
    
    try:
        response = llm.invoke(prompt)
        return extract_score(response.content)
    except Exception as e:
        print(f"   ⚠️ Faithfulness eval error: {str(e)[:30]}")
        return 0.5

def evaluate_relevancy(question, answer):
    """
    Evaluate how relevant the answer is to the question.
    Score: 0.0 - 1.0
    """
    prompt = f"""Rate how well this answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

On a scale 0-1:
- 1.0 = Directly and completely answers the question
- 0.75 = Answers well but missing some details
- 0.5 = Partially answers, covers main points
- 0.25 = Minimally relevant, mostly off-topic
- 0.0 = Completely irrelevant or refuses to answer

Return ONLY the numerical score (e.g., 0.85)"""
    
    try:
        response = llm.invoke(prompt)
        return extract_score(response.content)
    except Exception as e:
        print(f"   ⚠️ Relevancy eval error: {str(e)[:30]}")
        return 0.5

def evaluate_context_recall(question, contexts):
    """
    Evaluate if retrieved contexts contain information to answer the question.
    Score: 0.0 - 1.0
    """
    if not contexts or all(not c.strip() for c in contexts):
        return 0.0
    
    context_str = "\n\n".join(contexts)
    
    prompt = f"""Does the context contain enough information to answer the question?

QUESTION: {question}

CONTEXT: {context_str}

On a scale 0-1:
- 1.0 = Context fully covers all needed information
- 0.75 = Context covers most information needed
- 0.5 = Context covers some key information
- 0.25 = Context barely touches on the topic
- 0.0 = Context doesn't address the question at all

Return ONLY the numerical score (e.g., 0.85)"""
    
    try:
        response = llm.invoke(prompt)
        return extract_score(response.content)
    except Exception as e:
        print(f"   ⚠️ Context recall eval error: {str(e)[:30]}")
        return 0.5

def evaluate_answer_correctness(question, answer):
    """
    Evaluate if the answer is factually correct for parking system domain.
    Score: 0.0 - 1.0
    """
    prompt = f"""Evaluate if this answer about a parking booking system is factually correct and helpful.

QUESTION: {question}

ANSWER: {answer}

On a scale 0-1:
- 1.0 = Answer is accurate, complete, and helpful
- 0.75 = Answer is mostly correct with minor issues
- 0.5 = Answer is partially correct or incomplete
- 0.25 = Answer has significant errors or is unhelpful
- 0.0 = Answer is completely wrong or harmful

Return ONLY the numerical score (e.g., 0.85)"""
    
    try:
        response = llm.invoke(prompt)
        return extract_score(response.content)
    except Exception as e:
        print(f"   ⚠️ Correctness eval error: {str(e)[:30]}")
        return 0.5

# Main Evaluation Loop
print("🚀 Starting RAG system evaluation...\n")
print("="*80)

evaluation_results = {
    "questions": [],
    "answers": [],
    "contexts": [],
    "scores": {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "answer_correctness": [],
        "overall": []
    },
    "api_success_rate": 0,
    "metadata": {
        "total_questions": len(TEST_QUESTIONS),
        "backend_url": BACKEND_URL,
        "timestamp": datetime.now().isoformat()
    }
}

successful_queries = 0
failed_queries = 0

for i, question in enumerate(TEST_QUESTIONS):
    progress = (i + 1) / len(TEST_QUESTIONS) * 100
    print(f"\n[{i+1:2d}/{len(TEST_QUESTIONS)}] {progress:5.1f}% | Testing question...")
    print(f"❓ {question[:70]}")
    
    # Step 1: Query the RAG system
    result = query_rag_system(question)
    
    if not result:
        failed_queries += 1
        print("   ❌ Failed to get response from API")
        continue
    
    successful_queries += 1
    answer = result["answer"]
    contexts = result["contexts"]
    
    print(f"   ✓ Got answer ({len(answer)} chars)")
    print(f"   ✓ Retrieved {len(contexts)} context chunks")
    
    # Store data
    evaluation_results["questions"].append(question)
    evaluation_results["answers"].append(answer)
    evaluation_results["contexts"].append(contexts)
    
    # Step 2: Evaluate metrics
    print("   📊 Evaluating metrics...")
    
    faith_score = evaluate_faithfulness(question, answer, contexts)
    evaluation_results["scores"]["faithfulness"].append(faith_score)
    print(f"      • Faithfulness: {faith_score:.3f}")
    
    rel_score = evaluate_relevancy(question, answer)
    evaluation_results["scores"]["answer_relevancy"].append(rel_score)
    print(f"      • Relevancy: {rel_score:.3f}")
    
    recall_score = evaluate_context_recall(question, contexts)
    evaluation_results["scores"]["context_recall"].append(recall_score)
    print(f"      • Context Recall: {recall_score:.3f}")
    
    correct_score = evaluate_answer_correctness(question, answer)
    evaluation_results["scores"]["answer_correctness"].append(correct_score)
    print(f"      • Correctness: {correct_score:.3f}")
    
    # Calculate overall score
    overall = (faith_score + rel_score + recall_score + correct_score) / 4
    evaluation_results["scores"]["overall"].append(overall)
    print(f"      🎯 Overall: {overall:.3f}")
    
    # Rate limiting
    time.sleep(1)

# Calculate summary statistics
print("\n" + "="*80)
print("📊 EVALUATION SUMMARY")
print("="*80)

evaluation_results["api_success_rate"] = (successful_queries / len(TEST_QUESTIONS)) * 100

summary = {
    "total_questions": len(TEST_QUESTIONS),
    "successful_queries": successful_queries,
    "failed_queries": failed_queries,
    "api_success_rate": round(evaluation_results["api_success_rate"], 2)
}

print(f"\n📈 API Performance:")
print(f"   • Total Questions: {summary['total_questions']}")
print(f"   • Successful: {summary['successful_queries']}")
print(f"   • Failed: {summary['failed_queries']}")
print(f"   • Success Rate: {summary['api_success_rate']:.1f}%")

if successful_queries > 0:
    print(f"\n📊 RAG Quality Metrics:")
    
    for metric_name, scores in evaluation_results["scores"].items():
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            print(f"\n   {metric_name.replace('_', ' ').title()}:")
            print(f"      • Average: {avg_score:.3f}")
            print(f"      • Min: {min_score:.3f}")
            print(f"      • Max: {max_score:.3f}")
            
            summary[f"{metric_name}_avg"] = round(avg_score, 3)
            summary[f"{metric_name}_min"] = round(min_score, 3)
            summary[f"{metric_name}_max"] = round(max_score, 3)
    
    # Overall assessment
    overall_avg = summary.get("overall_avg", 0)
    print(f"\n🎯 Overall RAG System Score: {overall_avg:.3f}")
    
    if overall_avg >= 0.8:
        print("   ✅ Excellent performance!")
    elif overall_avg >= 0.6:
        print("   ✓ Good performance")
    elif overall_avg >= 0.4:
        print("   ⚠️ Needs improvement")
    else:
        print("   ❌ Poor performance - review system")

# Save detailed results
evaluation_results["summary"] = summary

output_file = f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Detailed results saved to: {output_file}")

# Save summary report
report_file = f"rag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("RAG SYSTEM EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Backend URL: {BACKEND_URL}\n\n")
    
    f.write("SUMMARY METRICS\n")
    f.write("-"*80 + "\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"📄 Summary report saved to: {report_file}")
print("\n✅ Evaluation Complete!")
print("="*80)