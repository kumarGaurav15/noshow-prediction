"""Flask API + simple staff dashboard. Run from the project root:  python -m app.api"""
import os
from flask import Flask, jsonify, request, Response
from noshow.predict import load_bundle, predict_one

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>No-show risk (decision support)</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#222}
label{display:block;margin-top:10px;font-size:14px}
input,select{width:100%;padding:7px;margin-top:3px;box-sizing:border-box}
button{margin-top:16px;padding:10px 18px;font-size:15px;cursor:pointer}
.card{border:1px solid #ccc;border-radius:8px;padding:14px;margin-top:18px}
.warn{color:#9a5b00}.err{color:#b00020}.small{font-size:13px;color:#555}
</style></head><body>
<h2>Appointment no-show risk</h2>
<p class="small">Decision support for human-reviewed reminders. Uses only information known at booking time.
Synthetic training data. Not for clinical decisions.</p>
<div id="form"></div><button onclick="run()">Estimate risk</button>
<div id="out"></div>
<script>
const today = new Date().toISOString().slice(0,10);
const fields = [
 ["booked_date","Booking date","date",today],
 ["appointment_date","Appointment date","date",today],
 ["appt_hour","Appointment hour (0-23)","number",10],
 ["clinic_type","Clinic type","select"],
 ["booking_channel","Booking channel (optional)","select"],
 ["sms_reminder_enabled","SMS reminder opt-in","yn"],
 ["is_follow_up","Follow-up visit","yn"],
 ["age_band","Age band","select"],
 ["distance_band","Distance band (optional)","select"],
 ["prior_appointments","Past appointments (completed before booking)","number",0],
 ["prior_no_shows","Of those, no-shows","number",0]];
let schema = {};
function build(){
  let h="";
  for(const [k,l,t,d] of fields){
    h+=`<label>${l}`;
    if(t==="select"){
      const opt = k.includes("channel")||k.includes("distance") ? '<option value="">(not provided)</option>' : '';
      h+=`<select id="${k}">${opt}${(schema[k]||[]).map(o=>`<option>${o}</option>`).join("")}</select>`;
    } else if(t==="yn"){
      h+=`<select id="${k}"><option value="1">Yes</option><option value="0">No</option></select>`;
    } else h+=`<input id="${k}" type="${t}" value="${d}">`;
    h+="</label>";
  }
  document.getElementById("form").innerHTML=h;
}
async function run(){
  const rec={};
  for(const [k,,t] of fields){
    const v=document.getElementById(k).value;
    if(v==="") continue;
    rec[k]=(t==="number"||t==="yn")?Number(v):v;
  }
  const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
  const d=await r.json(); const o=document.getElementById("out");
  if(!r.ok){o.innerHTML=`<div class="card err"><b>Please fix:</b><ul>${(d.errors||[d.error]).map(e=>`<li>${e}</li>`).join("")}</ul></div>`;return;}
  o.innerHTML=`<div class="card"><h3>Estimated no-show probability: ${(d.probability_no_show*100).toFixed(1)}%</h3>
   <p>Risk band: <b>${d.risk_band}</b> (review threshold ${(d.threshold*100).toFixed(0)}%)</p>
   <p><b>${d.recommendation}</b></p>
   <h4>Most influential factors</h4><ul>${d.top_factors.map(f=>`<li>${f.factor} (${f.value}): ${f.direction}, ${f.effect_pct_points>0?"+":""}${f.effect_pct_points} points</li>`).join("")}</ul>
   ${d.warnings.length?`<p class="warn"><b>Warnings:</b></p><ul class="warn">${d.warnings.map(w=>`<li>${w}</li>`).join("")}</ul>`:""}
   <p class="small">${d.uncertainty}</p><p class="small">${d.disclaimer}</p></div>`;
}
fetch("/schema").then(r=>r.json()).then(s=>{schema=s;build();});
</script></body></html>"""


def create_app(model_path=None):
    app = Flask(__name__)
    app.config["BUNDLE"] = load_bundle(model_path or os.getenv("NOSHOW_MODEL_PATH") or None)

    @app.get("/")
    def home():
        return Response(PAGE, mimetype="text/html")

    @app.get("/health")
    def health():
        m = app.config["BUNDLE"]["metadata"]
        return jsonify({"status": "ok", "model": m["model"], "sklearn": m["sklearn"]})

    @app.get("/schema")
    def schema():
        return jsonify(app.config["BUNDLE"]["categories"])

    @app.post("/predict")
    def predict():
        record = request.get_json(silent=True)
        if not isinstance(record, dict):
            return jsonify({"error": "Send a JSON object", "errors": ["Send a JSON object"]}), 400
        try:
            return jsonify(predict_one(app.config["BUNDLE"], record))
        except ValueError as e:
            errs = e.args[0] if e.args and isinstance(e.args[0], list) else [str(e)]
            return jsonify({"error": "invalid input", "errors": errs}), 400

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)