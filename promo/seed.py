import json, shutil, sys, os
from pathlib import Path
S = Path(sys.argv[1]); ROOT = S/"lectures"   # S holds site/assets/demo.json + demo-lecture.m4a (see README)
shutil.rmtree(ROOT, ignore_errors=True)
d = json.load(open(S/"site/assets/demo.json"))
W = d["words"]; dur = d["duration_ms"]
t0 = [w[1] for w in W]
t1 = [min(t0[i+1], t0[i]+700) if i+1 < len(t0) else dur for i in range(len(t0))]
sents = [{"sid": i+1, "w0": a, "wn": n, "t0": t0[a], "t1": t1[a+n-1]} for i,(a,n) in enumerate(d["sents"])]
tr = {"schema":1,"duration_ms":dur,"language":"en",
      "words":{"w":[w[0] for w in W],"t0":t0,"t1":t1,"p":[0.95]*len(W)},"sents":sents}
bul = [{"text":t,"cites":[],"grounded":0.9,"t_ms":ms,"tier":"sentence"} for t,ms in d["notes"]]
notes = {"schema":1,"engine":"qwen2.5:3b","summary":"","fallback":None,
  "sections":[{"heading":"Mitochondria","t0":0,"bullets":bul}],
  "terms":[],
  "quiz":[{"q":"What do mitochondria turn the energy stored in glucose into?","a":"ATP."},
          {"q":"Where does the electron transport chain sit?","a":"On the inner membrane."},
          {"q":"Roughly how much ATP does the third stage make per glucose?","a":"About 34."},
          {"q":"What are the ridges of the inner membrane called?","a":"Cristae."},
          {"q":"Where did mitochondria likely come from?","a":"Free-living bacteria that moved into early cells."},
          {"q":"What are the three stages of cellular respiration?","a":"Glycolysis, the Krebs cycle and electron transport."},
          {"q":"What should you read before next week?","a":"Chapter 7."}]}
def mk(folder, name, title, created, full=False, mins=50, words=9000):
    p = ROOT/folder/name; (p/".work").mkdir(parents=True)
    m = {"schema":1,"id":name,"title":title,"course":folder,"created_utc":created,"state":"complete",
         "stage_history":[],"source":{"kind":"microphone"},"recovered":False,"recovery_note":None,
         "audio":{"duration_ms": dur if full else mins*60000},
         "transcript":{"words": len(W) if full else words},"notes":{},"app":{"name":"Lectern","version":"0.1.1"}}
    (p/"lecture.json").write_text(json.dumps(m))
    if full:
        shutil.copy(S/"site/assets/demo-lecture.m4a", p/"audio.m4a")
        (p/"transcript.json").write_text(json.dumps(tr)); (p/"notes.json").write_text(json.dumps(notes))
        (p/"transcript.txt").write_text(" ".join(w[0] for w in W)); (p/"notes.md").write_text("# notes")
mk("BIO 101","2026-10-26_2328_intro-to-cell-biology-lecture-6","Intro to Cell Biology, Lecture 6","2026-10-26T23:28:00Z",True)
mk("BIO 101","2026-10-21_1002_cell-membranes","Week 5: Cell membranes","2026-10-21T14:02:00Z")
mk("CHEM 110","2026-10-23_0930_acids-and-bases","Acids and bases","2026-10-23T13:30:00Z")
mk("HIST 202","2026-10-22_1400_the-industrial-revolution","The Industrial Revolution","2026-10-22T18:00:00Z")
