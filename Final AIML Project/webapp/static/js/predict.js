let ALL_SYMPTOMS = [];

async function fetchFeatureNames(){
  try{
    const res = await fetch('/features');
    const data = await res.json();
    if(Array.isArray(data.features) && data.features.length){
      return data.features;
    }
  }catch(e){ /* ignore, fallback below */ }
  return null;
}

const COMMON_SYMPTOMS=[
  'fever','cough','headache','nausea','vomiting','fatigue','skin_rash','itching','shortness_of_breath','chest_pain','diarrhoea','sore_throat','loss_of_smell','loss_of_taste','runny_nose'
];

function titleCaseLabel(s){
  return (s||'')
    .toString()
    .replace(/[_-]+/g,' ')
    .split(' ')
    .filter(Boolean)
    .map(w=>w.charAt(0).toUpperCase()+w.slice(1).toLowerCase())
    .join(' ');
}

function renderSymptoms(list){
  const cont = document.getElementById('symptom-container');
  cont.innerHTML = '';
  (list||COMMON_SYMPTOMS).forEach(s=>{
    const id = 's_'+s.replace(/[^a-z0-9]/gi,'_');
    const div = document.createElement('label');
    div.className = 'symptom';
    div.innerHTML = `<input type="checkbox" value="${s}" id="${id}"><span>${titleCaseLabel(s)}</span>`;
    cont.appendChild(div);
  });
}

function getSelected(){
  return [...document.querySelectorAll('#symptom-container input:checked')].map(x=>x.value);
}

async function predict(){
  const btn = document.getElementById('predictBtn');
  btn.disabled = true; btn.textContent = 'Predicting...';
  try{
    const res = await fetch(window.SYMPTOMS_ENDPOINT,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symptoms:getSelected()})
    });
    const data = await res.json();
    const box = document.getElementById('result');
    if(!data.ok){
      box.classList.remove('hidden');
      box.innerHTML = `<h4>Error</h4><p>${data.error||'Something went wrong'}</p>`;
      return;
    }
    const tips = (data.tips||[]).map(t=>`<li>${t}</li>`).join('');
    const top5 = (data.top5||[]).map((r,i)=>{
      const pct = (r.confidence*100).toFixed(1)+'%';
      return `<tr><td>${i+1}</td><td>${r.disease}</td><td>${pct}</td></tr>`;
    }).join('');
    // Build modal content
    const modal = document.getElementById('resultModal');
    const content = document.getElementById('modalContent');
    const moreUrl = `https://www.google.com/search?q=${encodeURIComponent(data.disease+' site:mayoclinic.org OR site:webmd.com')}`;
    content.innerHTML = `
      <div class="card gradient" style="margin-bottom:1rem">
        <h3 style="margin:.2rem 0">${data.disease} <span class="badge">Prediction</span></h3>
        <p class='muted'>${data.description||''}</p>
      </div>
      <h4>Top 5 Predictions</h4>
      <div style="overflow:auto">
        <table class="table">
          <thead><tr><th>#</th><th>Disease</th><th>Confidence</th></tr></thead>
          <tbody>${top5}</tbody>
        </table>
      </div>
      <div style="display:flex;gap:.5rem;align-items:center;margin-top:.75rem">
        <button id="remediesBtn" class="btn btn-outline" type="button">Hide Remedies</button>
        <a class="btn btn-outline" target="_blank" rel="noopener" href="${moreUrl}">More info (WebMD/Mayo Clinic)</a>
      </div>
      <div id="remediesBox" class="glass" style="margin-top:.8rem;padding:.8rem">
        <h4 style="margin-top:0">Remedies / Precautions</h4>
        <ul>${tips}</ul>
      </div>
    `;
    modal.classList.remove('hidden');
    document.getElementById('modalClose').onclick = ()=> modal.classList.add('hidden');
    document.querySelector('#resultModal .modal-backdrop').onclick = ()=> modal.classList.add('hidden');
    const remediesBtn = document.getElementById('remediesBtn');
    const remediesBox = document.getElementById('remediesBox');
    if(remediesBtn && remediesBox){
      remediesBtn.addEventListener('click', ()=>{
        const isHidden = remediesBox.classList.contains('hidden');
        remediesBox.classList.toggle('hidden');
        remediesBtn.textContent = isHidden ? 'Hide Remedies' : 'Show Remedies';
      });
    }

    // Keep inline box as fallback
    box.classList.remove('hidden');
    box.innerHTML = `<h3>${data.disease}</h3><p class='muted'>${data.description||''}</p>`;
  }catch(e){
    alert('Prediction failed: '+e.message);
  }finally{
    btn.disabled = false; btn.textContent = 'Predict';
  }
}

function clearSel(){
  document.querySelectorAll('#symptom-container input:checked').forEach(c=>c.checked=false);
  const box = document.getElementById('result');
  box.classList.add('hidden');
  box.innerHTML = '';
}

document.addEventListener('DOMContentLoaded',async ()=>{
  const feats = await fetchFeatureNames();
  if(feats){
    // Normalize feature tokens: keep original keys but display prettified
    ALL_SYMPTOMS = feats.slice().sort((a,b)=>a.toLowerCase().localeCompare(b.toLowerCase()));
    renderSymptoms(ALL_SYMPTOMS);
  }else{
    ALL_SYMPTOMS = COMMON_SYMPTOMS.slice().sort((a,b)=>a.toLowerCase().localeCompare(b.toLowerCase()));
    renderSymptoms(ALL_SYMPTOMS);
  }
  document.getElementById('predictBtn').addEventListener('click', predict);
  document.getElementById('clearBtn').addEventListener('click', clearSel);
  const search = document.getElementById('symptomSearch');
  if(search){
    search.addEventListener('input', (e)=>{
      const q = e.target.value.trim().toLowerCase();
      if(!q){
        renderSymptoms(ALL_SYMPTOMS);
        return;
      }
      const filtered = ALL_SYMPTOMS.filter(s=>s.toLowerCase().includes(q) || s.replaceAll('_',' ').toLowerCase().includes(q));
      renderSymptoms(filtered);
    });
  }
});
