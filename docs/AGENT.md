## Objetivo
Construir **Drawdown Kill Switch** para MT5 en Windows: un “circuit breaker” distribuido en LAN que, ante un **drawdown instantáneo** superior a un umbral, ejecuta:

1) **Cierre inmediato** de todas las posiciones y eliminación de todas las órdenes pendientes.  
2) **Bloqueo de reapertura**: impedir que vuelvan a entrar órdenes (enforcement desde MT5 con un EA mínimo).  
3) **Propagación**: todas las terminales conectadas a la misma cuenta en otras máquinas aplican lo mismo.

El proyecto debe usar **Python mayoritariamente** y **MQL5 al mínimo** (solo un EA “Blocker” por terminal).

---

## Definiciones (terminología)
- **Cuenta**: identificador único por broker. Usar `account_key = "{ACCOUNT_LOGIN}@{ACCOUNT_SERVER}"` para evitar colisiones entre brokers.
- **Equity**: valor de cuenta en tiempo real.
- **Peak Equity**: máximo equity observado en una ventana definida (por defecto: “desde inicio del día”).
- **Drawdown instantáneo (DD)**:  
  - `DD_money = max(0, peak_equity - equity_actual)`  
  - `DD_pct = 100 * DD_money / peak_equity` (si `peak_equity > 0`)
- **LOCK**: estado distribuido por cuenta (locked/unlocked). Si `LOCK=1` entonces:
  - se liquida todo (flatten),
  - y se evita reapertura por enforcement (EA Blocker + watchdog).

---

## Alcance y no-alcance
### En alcance
- Coordinación LAN con un **Lock Server** (Python) que mantiene estado `LOCK` por cuenta.
- **Agent Python** por máquina (o al menos por máquina con capacidad de operar), que:
  - calcula DD,
  - activa el lock en el server,
  - ejecuta flatten usando la API Python de MT5,
  - y actúa como watchdog durante lock.
- **EA MQL5 mínimo** (“Blocker”) por terminal, que:
  - consulta el lock server,
  - si locked: cierra posiciones + borra pendientes,
  - ante cualquier nueva transacción mientras locked: vuelve a cerrar.
- Logs detallados y configuración por `.env`.

### No en alcance (por defecto)
- UI gráfica.
- Gestión compleja de usuarios/roles.
- Alta disponibilidad multi-server (se deja preparado para futuro).
- Integraciones cloud.

---

## Arquitectura de alto nivel
### Componentes
1) `lock_server` (Python, LAN)
- Autoridad del estado `LOCK` por cuenta.
- API HTTP simple (JSON).
- Persistencia opcional en SQLite (para reinicios).

2) `agent` (Python, por máquina)
- Se conecta al MT5 local (terminal instalado).
- Monitorea equity → calcula DD → dispara lock.
- Ejecuta flatten y watchdog.
- Reporta heartbeats al server (opcional).

3) `mql5_blocker` (EA mínimo, por terminal)
- Polling del lock server (HTTP WebRequest).
- Enforcement “dentro de MT5”: si locked, no hay exposición efectiva.

---

## Requisitos funcionales (acceptance criteria)
### Lock activation
- Si `DD_money >= THRESHOLD_MONEY` **o** `DD_pct >= THRESHOLD_PCT` ⇒ el agent debe:
  1) `POST /lock` con `account` (account_key), `reason`, `machine_id`.
  2) Ejecutar `flatten()` (posiciones + pendientes).
  3) Entrar en modo `LOCK` (watchdog).
- Reintentos: si el server no responde, el agent debe aplicar política configurable:
  - `FAIL_CLOSED=true`: asumir lock local y liquidar igual.
  - `FAIL_CLOSED=false`: no activar lock remoto, pero seguir intentándolo.

### Lock propagation
- Cuando una máquina activa lock, el resto:
  - debe ver `LOCK=1` con `GET /state?account=<account_key>`,
  - y ejecutar la misma política (flatten + enforcement).

### Unblock
- Debe existir `POST /unlock` protegido por API key (o token).
- Debe existir un `cooldown` configurable para evitar flapping:
  - `MIN_LOCK_SECONDS` (ej: 600s). Hasta que no se cumpla, `/unlock` debe rechazar o requerir `force=true`.

### Block de nuevas órdenes (enforcement)
- Durante `LOCK=1`:
  - Si aparece una nueva posición u orden pendiente por cualquier motivo (manual, copier, EA), debe ser cerrada/eliminada rápidamente por:
    - EA Blocker (primario),
    - Agent watchdog (secundario).

### “Cierre total”
- Debe cerrar:
  - todas las posiciones abiertas (todas las symbols),
  - todas las órdenes pendientes,
  - y dejar la cuenta “flat”.

---

## Requisitos no funcionales
- Windows 10/11.
- Python 3.11+ (x64).
- MT5 instalado y accesible localmente en cada máquina.
- Robustez:
  - Idempotencia: múltiples `POST /lock` no deben romper nada.
  - Backoff en polling para no saturar LAN.
- Observabilidad:
  - Logs en JSON (archivo rotativo) por agent.
  - Logs de auditoría en server (quién bloqueó, por qué, cuándo).

---

## Repo layout (propuesta)

/
AGENT.md
README.md
.env.example

server/
app.py
models.py
storage.py
auth.py
settings.py

agent/
main.py
settings.py
dd.py
mt5_client.py
flatten.py
watchdog.py
http_client.py
logging_setup.py

mql5/
Experts/
DD_KillSwitch_Blocker.mq5

tests/
test_dd.py
test_server_lock_state.py
test_agent_policy.py


---

## Configuración (.env)
Crear `.env.example` con estos campos:
- `LOCK_SERVER_URL=http://172.16.0.10:8765`
- `LOCK_API_KEY=change-me`
- `FAIL_CLOSED=true`
- `THRESHOLD_MONEY=300.0`
- `THRESHOLD_PCT=0.0`
- `BASELINE_MODE=daily_peak`  # daily_peak | session_peak
- `DAILY_RESET_HOUR=0`        # reset baseline (0 = medianoche local/broker)
- `POLL_INTERVAL_MS=250`
- `WATCHDOG_INTERVAL_MS=250`
- `MIN_LOCK_SECONDS=600`
- `MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe`  # opcional
- `MT5_LOGIN=` `MT5_PASSWORD=` `MT5_SERVER=`  # opcional (si se usa initialize con credenciales)
- `LOG_DIR=logs`

---

## Lock Server (Python)
### Stack recomendado
- FastAPI + Uvicorn
- Pydantic para modelos
- SQLite opcional para persistencia

### Modelo de estado
Clave primaria: `account_key` (string).
Campos:
- `locked: bool`
- `locked_since: epoch_seconds`
- `reason: str`
- `locked_by: machine_id`
- `lock_version: int` (incremental)
- `last_heartbeat: epoch_seconds` (opcional)

### Endpoints
- `GET /health` → 200 OK
- `GET /state?account=<account_key>`
  - Respuesta simple: `"1"` o `"0"` (para MQL5 sencillo)
  - y/o JSON opcional: `{ locked, locked_since, reason, lock_version }`
- `POST /lock` (auth required)
  - Body JSON:
    - `account` (required, account_key)
    - `reason` (required)
    - `machine_id` (required)
    - `ts` (optional)
  - Comportamiento:
    - Si ya locked: actualizar `reason` si viene, registrar auditoría, devolver 200.
    - Si no locked: set locked + locked_since + lock_version++.
- `POST /unlock` (auth required)
  - Body JSON:
    - `account` (required, account_key)
    - `machine_id`
    - `force` (default false)
  - Reglas:
    - si `now - locked_since < MIN_LOCK_SECONDS` y `force=false` ⇒ 409.
    - si unlocked ⇒ 200 idempotente.

### Auth
Usar header: `X-DKS-KEY: <LOCK_API_KEY>`.
El EA MQL5 NO envía key (por simplicidad); solo lee `/state` (public).
`/lock` y `/unlock` deben requerir key.

---

## Agent (Python)
### Responsabilidades
- Conectar a MT5 local.
- Calcular DD instantáneo.
- Disparar lock remoto al exceder umbral.
- Ejecutar flatten.
- Durante lock:
  - watchdog: cada `WATCHDOG_INTERVAL_MS` verificar y cerrar si reaparece exposición.
  - consultar `/state` para saber si sigue locked.

### Conexión MT5
Implementar wrapper `mt5_client.py`:
- `connect()`:
  - `mt5.initialize(path=MT5_PATH, login=..., password=..., server=...)` si se proveen.
  - si no, `mt5.initialize()` y leer `account_info()`.
- `account_login()` desde `account_info().login` y `account_server()` desde `account_info().server` para construir `account_key`.
- `equity()` desde `account_info().equity`.
- `positions()` con `positions_get()`.
- `orders()` con `orders_get()`.

### Cálculo de baseline (peak equity)
`BASELINE_MODE`:
- `daily_peak`: reset del peak 1 vez por día (hora configurable).
- `session_peak`: nunca resetea (desde arranque del agent).

Persistir peak en memoria; opcional: persistir en disco para reinicios.

### Disparo de lock (policy)
- Si `THRESHOLD_MONEY > 0` y `DD_money >= THRESHOLD_MONEY` ⇒ breach.
- Si `THRESHOLD_PCT > 0` y `DD_pct >= THRESHOLD_PCT` ⇒ breach.
- Si breach:
  1) intentar `POST /lock` (retry con backoff breve)
  2) `flatten()`
  3) set modo local locked (aunque server caiga, si `FAIL_CLOSED=true`)

### Flatten (core)
Implementar en `flatten.py`:
- Para cada posición:
  - enviar una operación DEAL opuesta (BUY→SELL / SELL→BUY) con:
    - `action=TRADE_ACTION_DEAL`
    - `symbol`, `volume`, `type`, `position=<ticket>`
    - `deviation`, `magic`, `comment="DKS flatten"`
- Para cada orden pendiente:
  - enviar `action=TRADE_ACTION_REMOVE`, `order=<ticket>`.

Reglas:
- Varias pasadas (hasta N=5) para asegurar limpieza.
- Loguear `retcode`, `comment`.
- Timeout total configurable (ej 3–5s).

### Watchdog durante lock
En `watchdog.py`:
- Loop:
  - si `/state` locked: `flatten()`
  - si unlocked: salir del modo watchdog.
- Frecuencia `WATCHDOG_INTERVAL_MS`.
- Evitar flood: si no hay exposición, dormir.

### Logging
- JSON logs con:
  - timestamp,
  - machine_id (hostname),
  - account_key,
  - dd_money, dd_pct,
  - lock_state,
  - acciones (flatten, lock request, unlock detected),
  - errores MT5 (last_error).

---

## EA MQL5 mínimo (Blocker)
Archivo: `mql5/Experts/DD_KillSwitch_Blocker.mq5`

### Requisitos
- NO calcula DD.
- SOLO consulta el lock server y aplica enforcement.

### Comportamiento
- OnInit:
  - set timer milisegundos (ej 500ms).
- OnTimer:
  - `GET /state?account=<login@server>`:
    - si responde `"1"`: `locked=true` y ejecutar `CloseAll()`.
- OnTradeTransaction:
  - si `locked=true`: ejecutar `CloseAll()`.

### CloseAll()
- Cerrar todas las posiciones (PositionClose por ticket).
- Eliminar pendientes (OrderDelete por ticket).
- Varias pasadas opcional.

### Config y operación
- El usuario debe permitir `WebRequest()` hacia el server en:
  - Tools → Options → Expert Advisors → “Allow WebRequest for listed URL”.

### Fail-safe (opcional)
Configurable en input:
- `FAIL_CLOSED_ON_HTTP_ERROR`:
  - si true: si WebRequest falla ⇒ asumir locked.
  - default: false.

---

## Tests
### Unit tests (sin MT5 real)
- Mock del módulo `MetaTrader5` para:
  - `account_info()`, `positions_get()`, `orders_get()`, `order_send()`.
- Tests:
  - dd 계산: peak update, reset diario, %.
  - policy: breach por money/pct.
  - server: lock/unlock idempotente, cooldown.
  - agent: cuando locked, watchdog llama flatten.

### Smoke test (manual)
- Iniciar `server`.
- Iniciar `agent` en 1 máquina.
- Forzar `POST /lock` con curl → verificar:
  - EA en todas las terminales cierra todo.
- Forzar `POST /unlock` → verificar:
  - EA deja de cerrar (no enforcement).
- Forzar reapertura manual durante lock → debe cerrarse inmediatamente.

---

## “Definition of Done”
- `server` corre en LAN, mantiene lock por cuenta, y expone `/state`, `/lock`, `/unlock`.
- `agent` corre en Windows y:
  - detecta breach por DD,
  - activa lock,
  - ejecuta flatten de posiciones + pendientes,
  - watchdog durante lock.
- `EA Blocker` en cada terminal:
  - consulta `/state` y aplica enforcement.
- Logs y `.env.example` completos.
- Tests unitarios pasan con mocks.
- README con pasos de instalación y operación (Windows + MT5).

---

## Implementación: orden sugerido
1) Implementar `server` (FastAPI) + estado en memoria + auth + endpoints.
2) Implementar `agent` con:
   - settings + logging
   - mt5_client wrapper
   - dd + baseline
   - flatten
   - lock/unlock polling
3) Implementar `EA Blocker` mínimo.
4) Tests + README.
