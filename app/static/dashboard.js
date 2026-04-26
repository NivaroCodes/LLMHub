const el = (id) => document.getElementById(id);

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function money(value) {
  return `$${Number(value || 0).toFixed(4)}`;
}

function latency(value) {
  return `${Math.round(Number(value || 0))} ms`;
}

function setKpis(kpis) {
  el("kpi-total").textContent = `${kpis.total_requests ?? 0}`;
  el("kpi-success").textContent = percent(kpis.success_rate);
  el("kpi-p95").textContent = latency(kpis.p95_latency_ms);
  el("kpi-cache").textContent = percent(kpis.cache_hit_rate);
  el("kpi-fallback").textContent = percent(kpis.fallback_rate);
  el("kpi-cost").textContent = money(kpis.cost_usd);
}

function setProviders(rows) {
  const body = el("providers-table");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="5">Нет данных</td></tr>';
    return;
  }
  body.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${row.provider}</td>
        <td>${row.requests}</td>
        <td>${row.errors}</td>
        <td>${latency(row.avg_latency_ms)}</td>
        <td>${money(row.cost_usd)}</td>
      </tr>`
    )
    .join("");
}

function setFailures(items) {
  const list = el("failures-list");
  if (!items.length) {
    list.innerHTML = "<li>Ошибок за последнее время нет</li>";
    return;
  }
  list.innerHTML = items
    .map(
      (item) => `
      <li>
        <div class="failure-time">${new Date(item.timestamp).toLocaleString()}</div>
        <div><strong>${item.provider}/${item.model}</strong> · ${item.latency_ms} ms</div>
        <div>${item.error || "unknown error"}</div>
      </li>`
    )
    .join("");
}

function setBars(points) {
  const bars = el("bars");
  if (!points.length) {
    bars.innerHTML = "<p>Недостаточно данных для графика</p>";
    return;
  }

  const maxReq = Math.max(...points.map((p) => p.requests), 1);
  const maxErr = Math.max(...points.map((p) => p.errors), 1);
  bars.innerHTML = points
    .map((point) => {
      const reqH = Math.max(4, Math.round((point.requests / maxReq) * 130));
      const errH = point.errors > 0 ? Math.max(3, Math.round((point.errors / maxErr) * 40)) : 0;
      const title = `${new Date(point.bucket_start).toLocaleTimeString()} · req=${point.requests}, err=${point.errors}, p95=${point.p95_latency_ms}ms`;
      return `
        <div class="bar-wrap" title="${title}">
          <span class="bar requests" style="height:${reqH}px"></span>
          <span class="bar errors" style="height:${errH}px"></span>
        </div>
      `;
    })
    .join("");
}

async function loadDashboard() {
  const windowMinutes = Number(el("window").value || 60);
  const bucketMinutes = windowMinutes <= 60 ? 2 : windowMinutes <= 180 ? 5 : 10;

  const [overviewRes, seriesRes, failuresRes] = await Promise.all([
    fetch(`/monitoring/overview?window_minutes=${windowMinutes}`),
    fetch(`/monitoring/timeseries?window_minutes=${windowMinutes}&bucket_minutes=${bucketMinutes}`),
    fetch("/monitoring/failures?limit=20"),
  ]);

  if (!overviewRes.ok || !seriesRes.ok || !failuresRes.ok) {
    throw new Error("monitoring api unavailable");
  }

  const overview = await overviewRes.json();
  const series = await seriesRes.json();
  const failures = await failuresRes.json();

  setKpis(overview.kpis || {});
  setProviders(overview.providers || []);
  setBars(series.points || []);
  setFailures(failures.items || []);
  el("updated-at").textContent = `Обновлено: ${new Date().toLocaleTimeString()}`;
}

async function safeLoad() {
  try {
    await loadDashboard();
  } catch {
    el("updated-at").textContent = "Ошибка загрузки мониторинга";
  }
}

el("refresh").addEventListener("click", safeLoad);
el("window").addEventListener("change", safeLoad);

safeLoad();
setInterval(safeLoad, 15000);
