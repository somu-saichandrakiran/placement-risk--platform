# test_api.py
import requests

url = "http://localhost:8000/v1/predict"

payload = {
    "tenth_percent":       78.5,
    "twelfth_percent":     72.0,
    "cgpa":                67.5,
    "employability_score": 74.0,
    "mba_percent":         62.0,
    "internship_count":    1,
    "project_count":       3,
    "hackathon_count":     1,
    "active_backlogs":     0,
    "work_exp_flag":       0,
    "specialization_12th": "Science",
    "degree_type":         "Sci&Tech",
    "specialization":      "Mkt&Fin",
    "college_tier":        "T2"
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Status: {response.status_code}")
print(f"\nPlacement probability : {data['placement']['probability']:.1%}")
print(f"Confidence interval   : {data['placement']['confidence_interval']}")
print(f"Salary P50            : ₹{data['salary']['p50_inr']:,}")
print(f"Salary range          : ₹{data['salary']['p10_inr']:,} – ₹{data['salary']['p90_inr']:,}")
print(f"\nTop drivers:")
for d in data['top_drivers']:
    print(f"  {d['direction'].upper():7} {d['feature']:<25} impact={d['impact_pct']:.1f}%")
print(f"\nTop recommendations:")
for r in data['top_recommendations']:
    print(f"  {r['action']:<30} gain={r['probability_gain']:+.1%}")
print(f"\nLatency : {data['latency_ms']}ms")
print(f"Model   : {data['model_version']}")
print(f"\nDisclaimer: {data['bias_disclaimer']}")