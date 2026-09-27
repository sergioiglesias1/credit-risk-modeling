const form = document.getElementById('score-form');
const result = document.getElementById('result');

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const usd = v => v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    el.removeAttribute('aria-invalid');
    if (el.type === 'number') payload[el.name] = el.value === '' ? null : Number(el.value);
    else if (el.name === 'emp_length_years') payload[el.name] = el.value === '' ? null : Number(el.value);
    else if (el.name === 'job_title') payload[el.name] = el.value.trim() || null;
    else payload[el.name] = el.value;
  }

  const button = form.querySelector('button');
  button.disabled = true;
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) return showErrors(data.detail);
    showResult(data);
  } catch {
    result.innerHTML = '<p class="error">Request failed. Try again.</p>';
  } finally {
    button.disabled = false;
  }
});

function showResult(r) {
  const op = r.decision === 'approve' ? '&lt;' : '&ge;';
  result.innerHTML = `
    <div class="table-wrap"><table class="kv"><tbody>
      <tr><th>Probability of default</th><td class="num">${pct(r.pd)}</td></tr>
      <tr><th>Loss given default</th><td class="num">${pct(r.lgd)}</td></tr>
      <tr><th>Exposure at default</th><td class="num">${usd(r.ead)}</td></tr>
      <tr><th>Expected loss</th><td class="num">${usd(r.expected_loss)} &middot; ${pct(r.expected_loss_pct, 2)}</td></tr>
      <tr><th>Decision</th><td class="num decision ${r.decision}">${r.decision === 'approve' ? 'Approve' : 'Reject'}
        <span class="note">(PD ${op} ${pct(r.threshold)})</span></td></tr>
    </tbody></table></div>`;
}

function showErrors(detail) {
  const items = (Array.isArray(detail) ? detail : [{ msg: String(detail), loc: [] }]).map(d => {
    const field = d.loc[d.loc.length - 1];
    const input = form.elements[field];
    if (input) input.setAttribute('aria-invalid', 'true');
    const label = input ? input.closest('label').firstChild.textContent.trim() : field;
    return `<li>${label}: ${d.msg}</li>`;
  });
  result.innerHTML = `<ul class="error">${items.join('')}</ul>`;
}
