# Guía 2: Orquestación de Entrenamiento/Inferencia con n8n

## Objetivo

Crear workflows en n8n que orquesten procesos de machine learning: entrenamiento programado, inferencia bajo demanda, logging de métricas y notificaciones automáticas.

## Casos de Uso

1. **Reentrenamiento Programado:** Entrenar modelo semanalmente con datos nuevos
2. **Inferencia Bajo Demanda:** Endpoint de predicción vía webhook
3. **Monitoreo de Performance:** Detectar drift y alertar

## Workflow 1: Entrenamiento Programado

### Arquitectura

```
[Cron Trigger: Weekly]
    ↓
[HTTP: Fetch New Data from API/DB]
    ↓
[Code: Validate Data Quality]
    ↓
[HTTP: POST to FastAPI /train endpoint]
    ↓
[Wait for Training Completion]
    ↓
[HTTP: GET Training Metrics]
    ↓
[Code: Compare with Previous Model]
    ↓
[Branch: Better/Worse]
    ↓ (better)                    ↓ (worse/equal)
[HTTP: Deploy New Model]      [Log: Keep Old Model]
    ↓                              ↓
[Google Sheets: Log Metrics]  [Google Sheets: Log Metrics]
    ↓                              ↓
[Slack: Success Notification] [Slack: No Deployment Needed]
```

### Implementación

#### Nodo 1: Cron Trigger
- **Trigger:** Cron
- **Mode:** Weekly
- **Day:** Sunday
- **Hour:** 2
- **Minute:** 0

#### Nodo 2: Fetch New Data
**HTTP Request:**
```json
{
  "method": "GET",
  "url": "https://api.ejemplo.com/data/latest",
  "authentication": "headerAuth",
  "sendHeaders": true,
  "headerParameters": {
    "parameters": [
      {
        "name": "Authorization",
        "value": "Bearer {{$credentials.apiToken}}"
      }
    ]
  },
  "options": {
    "timeout": 30000
  }
}
```

#### Nodo 3: Validate Data
**Code (JavaScript):**
```javascript
const data = $input.all();
const records = data[0].json.records;

// Validaciones
const errors = [];

if (!records || records.length === 0) {
  errors.push('No records found');
}

if (records.length < 1000) {
  errors.push(`Insufficient data: ${records.length} records (min: 1000)`);
}

// Verificar columnas requeridas
const requiredColumns = ['feature1', 'feature2', 'target'];
const sampleRecord = records[0];
const missingColumns = requiredColumns.filter(col => !(col in sampleRecord));

if (missingColumns.length > 0) {
  errors.push(`Missing columns: ${missingColumns.join(', ')}`);
}

// Si hay errores, retornar para manejo
if (errors.length > 0) {
  return [{
    json: {
      valid: false,
      errors: errors,
      record_count: records.length
    }
  }];
}

// Datos válidos
return [{
  json: {
    valid: true,
    records: records,
    record_count: records.length,
    validation_timestamp: new Date().toISOString()
  }
}];
```

#### Nodo 4: IF Node - Data Valid?
- **Condition:** `{{ $json.valid }} === true`
- **Si false:** ir a Error Handler

#### Nodo 5: Train Model via API
**HTTP Request:**
```json
{
  "method": "POST",
  "url": "http://localhost:8000/api/v1/train",
  "sendBody": true,
  "bodyParameters": {
    "data": "={{ $json.records }}",
    "model_type": "xgboost",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 5,
      "learning_rate": 0.1
    }
  },
  "options": {
    "timeout": 600000,
    "response": {
      "response": {
        "fullResponse": true
      }
    }
  }
}
```

**Response esperada:**
```json
{
  "job_id": "train_20250107_123456",
  "status": "running",
  "eta_seconds": 300
}
```

#### Nodo 6: Wait for Completion
**Code (JavaScript) con polling:**
```javascript
const jobId = $json.job_id;
const apiUrl = `http://localhost:8000/api/v1/jobs/${jobId}`;

// Polling con timeout
const maxWaitTime = 600000; // 10 minutos
const pollInterval = 10000; // 10 segundos
const startTime = Date.now();

async function checkStatus() {
  const response = await $http.get(apiUrl);
  return response.data;
}

let status = 'running';
let result = null;

while (status === 'running' && (Date.now() - startTime) < maxWaitTime) {
  result = await checkStatus();
  status = result.status;
  
  if (status === 'running') {
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
}

if (status === 'completed') {
  return [{
    json: {
      success: true,
      job_id: jobId,
      metrics: result.metrics,
      model_path: result.model_path
    }
  }];
} else if (status === 'running') {
  return [{
    json: {
      success: false,
      error: 'Training timeout',
      job_id: jobId
    }
  }];
} else {
  return [{
    json: {
      success: false,
      error: result.error,
      job_id: jobId
    }
  }];
}
```

#### Nodo 7: Compare with Previous Model
**Code:**
```javascript
const currentMetrics = $json.metrics;

// Obtener métricas del modelo anterior desde Google Sheets o DB
const previousMetrics = await $http.get('http://localhost:8000/api/v1/models/latest/metrics');

const improvement = {
  accuracy: currentMetrics.accuracy - previousMetrics.data.accuracy,
  f1_score: currentMetrics.f1_score - previousMetrics.data.f1_score,
  roc_auc: currentMetrics.roc_auc - previousMetrics.data.roc_auc
};

const shouldDeploy = improvement.accuracy > 0.01 || improvement.f1_score > 0.01;

return [{
  json: {
    current_metrics: currentMetrics,
    previous_metrics: previousMetrics.data,
    improvement: improvement,
    should_deploy: shouldDeploy,
    model_path: $json.model_path
  }
}];
```

#### Nodo 8: IF Node - Deploy?
- **Condition:** `{{ $json.should_deploy }} === true`

#### Nodo 9A: Deploy Model (si mejora)
**HTTP Request:**
```json
{
  "method": "POST",
  "url": "http://localhost:8000/api/v1/models/deploy",
  "sendBody": true,
  "bodyParameters": {
    "model_path": "={{ $json.model_path }}",
    "environment": "production"
  }
}
```

#### Nodo 9B: Keep Old Model (si no mejora)
**Code:**
```javascript
return [{
  json: {
    action: 'no_deployment',
    reason: 'New model did not improve significantly',
    improvement: $json.improvement
  }
}];
```

#### Nodo 10: Log to Google Sheets
**Google Sheets:**
- **Operation:** Append
- **Sheet:** "Training_Logs"
- **Columns:**
  - timestamp: `{{ $now.toISO() }}`
  - model_id: `{{ $json.job_id }}`
  - accuracy: `{{ $json.current_metrics.accuracy }}`
  - f1_score: `{{ $json.current_metrics.f1_score }}`
  - deployed: `{{ $json.should_deploy }}`
  - improvement: `{{ JSON.stringify($json.improvement) }}`

#### Nodo 11: Slack Notification
**Slack:**
- **Channel:** #ml-ops
- **Message:**
```
🤖 Model Training Complete

Job ID: {{ $json.job_id }}
Status: {{ $json.should_deploy ? '✅ Deployed' : '⏸️ Not Deployed' }}

Metrics:
• Accuracy: {{ ($json.current_metrics.accuracy * 100).toFixed(2) }}%
• F1 Score: {{ ($json.current_metrics.f1_score * 100).toFixed(2) }}%
• ROC-AUC: {{ ($json.current_metrics.roc_auc * 100).toFixed(2) }}%

Improvement:
• Accuracy: {{ ($json.improvement.accuracy * 100).toFixed(2) }}%
• F1: {{ ($json.improvement.f1_score * 100).toFixed(2) }}%

Time: {{ $now.toFormat('yyyy-MM-dd HH:mm:ss') }}
```

#### Error Handler
**En todos los nodos críticos, añadir "On Error" connection:**

**Error Handler Node (Code):**
```javascript
const error = $json.error || $input.item.json.error;
const nodeName = $workflow.active ? $node.name : 'Unknown';

return [{
  json: {
    workflow: 'ML Training Pipeline',
    node: nodeName,
    error_message: error.message || error,
    timestamp: new Date().toISOString(),
    input_data: JSON.stringify($input.all())
  }
}];
```

**Conectar a Slack:**
```
❌ Training Pipeline Failed

Node: {{ $json.node }}
Error: {{ $json.error_message }}
Time: {{ $json.timestamp }}

Please check the logs for details.
```

## Workflow 2: Inferencia Bajo Demanda

### Arquitectura Simplificada

```
[Webhook Trigger]
    ↓
[Validate Input]
    ↓
[HTTP: POST to Model API]
    ↓
[Format Response]
    ↓
[Return to Client]
    ↓
[Log Request (async)]
```

### Nodo 1: Webhook
- **Method:** POST
- **Path:** `/webhook/predict`
- **Authentication:** Header Auth (API Key)
- **Response Mode:** Last Node

### Nodo 2: Validate Input
```javascript
const input = $json.body;

if (!input.features || !Array.isArray(input.features)) {
  return [{
    json: {
      error: 'Invalid input: features array required',
      status: 400
    }
  }];
}

if (input.features.length !== 10) {  // Ejemplo: 10 features esperados
  return [{
    json: {
      error: `Expected 10 features, got ${input.features.length}`,
      status: 400
    }
  }];
}

return [{
  json: {
    valid: true,
    features: input.features,
    request_id: `req_${Date.now()}`
  }
}];
```

### Nodo 3: Call Model API
```json
{
  "method": "POST",
  "url": "http://localhost:8000/api/v1/predict",
  "sendBody": true,
  "bodyParameters": {
    "features": "={{ $json.features }}",
    "request_id": "={{ $json.request_id }}"
  }
}
```

### Nodo 4: Format Response
```javascript
const prediction = $json.prediction;
const probability = $json.probability;

return [{
  json: {
    request_id: $('Validate Input').item.json.request_id,
    prediction: prediction,
    probability: probability,
    timestamp: new Date().toISOString()
  }
}];
```

### Nodo 5: Respond to Webhook
**Webhook Response:**
- **Response Code:** 200
- **Response Data:** `={{ $json }}`

### Nodo 6: Log to Sheets (async, no afecta respuesta)
```json
{
  "timestamp": "={{ $json.timestamp }}",
  "request_id": "={{ $json.request_id }}",
  "prediction": "={{ $json.prediction }}",
  "probability": "={{ $json.probability }}"
}
```

## Testing de Workflows

### Test con cURL
```bash
# Test webhook de inferencia
curl -X POST http://localhost:5678/webhook/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "features": [0.5, 1.2, 0.8, 1.5, 0.3, 0.9, 1.1, 0.7, 1.3, 0.6]
  }'
```

## Mejores Prácticas

1. **Timeouts apropiados** para operaciones largas
2. **Retry logic** en llamadas HTTP
3. **Logging completo** de cada paso
4. **Error handling** en todos los nodos
5. **Validación de datos** antes de procesamiento
6. **Credenciales seguras** (nunca hardcodear)
7. **Monitoring** con Google Sheets o DB

## Recursos

- [n8n HTTP Request Node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)
- [n8n Webhook Node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/)
- [n8n Code Node Examples](https://docs.n8n.io/code-examples/)

---

**Siguiente:** [Guía 3: Pipeline RAG con n8n](./guia-3-pipeline-rag.md)
