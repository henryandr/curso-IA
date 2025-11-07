# Guía 1: Workflow n8n de ETL Básico

## Objetivo

Crear un workflow en n8n que automatice la extracción, transformación y carga (ETL) de datos desde Google Sheets o CSV, realice limpieza básica y almacene los resultados.

## Prerequisitos

- [ ] n8n instalado (Docker o Desktop app)
- [ ] Cuenta de Google (si usas Google Sheets)
- [ ] Conocimientos básicos de n8n (nodes, connections)

## Caso de Uso

**Escenario:** Tienes un equipo que registra datos de ventas en Google Sheets. Quieres automatizar la limpieza y almacenamiento de estos datos en una base de datos o archivo CSV procesado.

## Arquitectura del Workflow

```
[Trigger: Schedule/Manual]
    ↓
[Google Sheets / HTTP Request - Read CSV]
    ↓
[Code: Data Cleaning]
    ↓
[Code: Data Validation]
    ↓
[Branch: Valid/Invalid Data]
    ↓ (valid)                ↓ (invalid)
[Store Clean Data]       [Log Errors]
    ↓                        ↓
[Update Google Sheets]   [Notify via Slack]
    ↓
[Success Notification]
```

## Implementación Paso a Paso

### Paso 1: Crear Workflow y Configurar Trigger

1. Abre n8n y crea un nuevo workflow
2. Nombra el workflow: "ETL - Sales Data Pipeline"
3. Añade un nodo **Schedule Trigger**
   - **Mode:** Every Day
   - **Hour:** 2 (2 AM)
   - **Minute:** 0
   - Esto ejecutará el workflow diariamente a las 2 AM

**Alternativa:** Para testing, añade también un **Manual Trigger** en paralelo

### Paso 2: Extraer Datos (Extract)

#### Opción A: Desde Google Sheets

1. Añade nodo **Google Sheets**
2. Configura credenciales:
   - Click en "Credentials" → "Create New"
   - Sigue el flujo de OAuth de Google
3. Configuración del nodo:
   - **Operation:** Read
   - **Document ID:** [ID de tu Google Sheet]
   - **Sheet Name:** "Sales_Raw"
   - **Range:** A1:Z (todas las columnas)
   - **Options → RAW Data:** Desactivado (queremos objetos JSON)

#### Opción B: Desde CSV en URL

1. Añade nodo **HTTP Request**
2. Configuración:
   - **Method:** GET
   - **URL:** https://ejemplo.com/sales_data.csv
   - **Response Format:** String
3. Añade nodo **Code** para parsear CSV:

```javascript
// Parse CSV to JSON
const csvData = $input.item.json.data;
const lines = csvData.split('\n');
const headers = lines[0].split(',');

const parsedData = lines.slice(1).map(line => {
  const values = line.split(',');
  return headers.reduce((obj, header, index) => {
    obj[header.trim()] = values[index]?.trim();
    return obj;
  }, {});
});

return parsedData.map(item => ({ json: item }));
```

### Paso 3: Transformar Datos (Transform) - Limpieza

Añade nodo **Code (Python o JavaScript)**

**JavaScript:**
```javascript
// Data Cleaning
const items = $input.all();

const cleanedItems = items.map(item => {
  const data = item.json;
  
  // 1. Limpiar espacios en blanco
  Object.keys(data).forEach(key => {
    if (typeof data[key] === 'string') {
      data[key] = data[key].trim();
    }
  });
  
  // 2. Convertir tipos de datos
  data.amount = parseFloat(data.amount) || 0;
  data.quantity = parseInt(data.quantity) || 0;
  
  // 3. Normalizar fechas
  if (data.date) {
    data.date = new Date(data.date).toISOString().split('T')[0];
  }
  
  // 4. Normalizar categorías
  if (data.category) {
    data.category = data.category.toLowerCase();
  }
  
  // 5. Eliminar campos vacíos
  const cleaned = {};
  Object.keys(data).forEach(key => {
    if (data[key] !== null && data[key] !== undefined && data[key] !== '') {
      cleaned[key] = data[key];
    }
  });
  
  return { json: cleaned };
});

return cleanedItems;
```

**Python (más potente para transformaciones complejas):**
```python
import pandas as pd
from datetime import datetime

# Obtener items de entrada
items = _input.all()

# Convertir a DataFrame para limpieza masiva
df = pd.DataFrame([item['json'] for item in items])

# 1. Limpiar espacios
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# 2. Convertir tipos
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

# 3. Normalizar fechas
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# 4. Normalizar categorías
if 'category' in df.columns:
    df['category'] = df['category'].str.lower()

# 5. Eliminar duplicados
df = df.drop_duplicates()

# 6. Rellenar nulos estratégicamente
df['category'].fillna('unknown', inplace=True)

# Convertir de vuelta a formato n8n
return [{'json': row.to_dict()} for _, row in df.iterrows()]
```

### Paso 4: Validar Datos

Añade nodo **Code** para validación:

```javascript
// Data Validation
const items = $input.all();

const validItems = [];
const invalidItems = [];

items.forEach(item => {
  const data = item.json;
  const errors = [];
  
  // Validaciones
  if (!data.date || isNaN(new Date(data.date))) {
    errors.push('Invalid date');
  }
  
  if (data.amount <= 0) {
    errors.push('Amount must be positive');
  }
  
  if (data.quantity <= 0) {
    errors.push('Quantity must be positive');
  }
  
  if (!data.product_id) {
    errors.push('Product ID is required');
  }
  
  // Clasificar item
  if (errors.length === 0) {
    validItems.push({ json: data });
  } else {
    invalidItems.push({ 
      json: { 
        ...data, 
        validation_errors: errors,
        validation_timestamp: new Date().toISOString()
      } 
    });
  }
});

// Retornar ambos conjuntos con metadata
return [
  {
    json: {
      valid_count: validItems.length,
      invalid_count: invalidItems.length,
      valid_items: validItems.map(i => i.json),
      invalid_items: invalidItems.map(i => i.json)
    }
  }
];
```

### Paso 5: Bifurcar Flujo (Valid vs Invalid)

1. Añade nodo **Split Out** después de la validación
   - **Field to Split Out:** valid_items
2. Añade otro nodo **Split Out** en paralelo
   - **Field to Split Out:** invalid_items

### Paso 6: Almacenar Datos Válidos (Load)

#### Opción A: Google Sheets

1. Añade nodo **Google Sheets** (en la rama de válidos)
2. Configuración:
   - **Operation:** Append
   - **Document ID:** [Tu Sheet ID]
   - **Sheet Name:** "Sales_Clean"
   - **Options:**
     - **Data Mode:** Auto-map Input Data
     - **Value Input Mode:** User Entered

#### Opción B: Escribir a CSV

1. Añade nodo **Code** para generar CSV:

```javascript
const items = $input.all();

// Generar CSV
const headers = Object.keys(items[0].json);
const csvContent = [
  headers.join(','),
  ...items.map(item => 
    headers.map(h => item.json[h]).join(',')
  )
].join('\n');

return [{ 
  json: { 
    filename: `sales_clean_${new Date().toISOString().split('T')[0]}.csv`,
    content: csvContent
  } 
}];
```

2. Añade nodo **Write File** o **HTTP Request** para guardar

#### Opción C: Base de Datos PostgreSQL

1. Añade nodo **PostgreSQL**
2. Configuración:
   - **Operation:** Insert
   - **Table:** sales_clean
   - **Columns:** Auto-map desde input

### Paso 7: Manejar Datos Inválidos

En la rama de datos inválidos:

1. Añade nodo **Google Sheets** o **Write File**
   - Guarda en sheet "Sales_Errors" o archivo de errores
   
2. Añade nodo **Slack** para notificación:
   - **Channel:** #data-alerts
   - **Message:**
   ```
   ⚠️ ETL Pipeline - Data Validation Errors
   
   Total errors: {{ $json.invalid_items.length }}
   Timestamp: {{ new Date().toISOString() }}
   
   Please review the Sales_Errors sheet for details.
   ```

### Paso 8: Notificación de Éxito

En la rama de datos válidos, después de guardar:

1. Añade nodo **Slack**:
   - **Channel:** #data-updates
   - **Message:**
   ```
   ✅ ETL Pipeline - Success
   
   Processed: {{ $('Code').item.json.valid_count }} records
   Errors: {{ $('Code').item.json.invalid_count }} records
   Date: {{ new Date().toISOString().split('T')[0] }}
   ```

### Paso 9: Manejo de Errores Global

1. Click en el workflow (background)
2. En **Settings** → **Error Workflow:**
   - Activa "Continue on Fail" en nodes críticos
   - Añade connections "On Error"

**Error Handler Node:**
```javascript
// Error Handler Code
const error = $json.error;
const nodeName = $json.node;

return [{
  json: {
    error_message: error.message,
    error_node: nodeName,
    timestamp: new Date().toISOString(),
    workflow_id: $workflow.id
  }
}];
```

Conecta a **Slack** para notificar errores críticos.

## Configuración de Credenciales

### Google Sheets

1. n8n → Credentials → Google Sheets API
2. Sigue OAuth flow
3. Permisos: Read and Write Spreadsheets

### Slack

1. Crea Slack App en api.slack.com
2. Añade OAuth scope: `chat:write`
3. Instala app en workspace
4. Copia token en n8n credentials

## Testing

1. **Test Manual:** Click en "Execute Workflow" con Manual Trigger
2. **Verifica:**
   - Datos extraídos correctamente
   - Limpieza aplicada
   - Validación funciona (prueba con datos buenos y malos)
   - Datos guardados en destino
   - Notificaciones enviadas
3. **Review Execution:** Revisa cada nodo para ver datos intermedios

## Monitoreo y Logging

Añade nodo **Code** al final:

```javascript
// Log Execution Summary
const summary = {
  workflow_name: 'ETL - Sales Data',
  execution_id: $execution.id,
  timestamp: new Date().toISOString(),
  records_processed: $('Code').item.json.valid_count,
  records_failed: $('Code').item.json.invalid_count,
  duration_ms: Date.now() - new Date($execution.startedAt).getTime()
};

// Opcional: Guardar en Google Sheets "ETL_Logs"
return [{ json: summary }];
```

## Mejoras Avanzadas

### 1. Incrementalidad
Añade lógica para procesar solo nuevos registros:

```javascript
// En nodo de extracción
const lastProcessedDate = await getLastProcessedDate(); // desde config
const newRecords = allRecords.filter(r => 
  new Date(r.date) > new Date(lastProcessedDate)
);
```

### 2. Retry Logic
En nodes críticos:
- Settings → **Retry On Fail:** Yes
- **Max Tries:** 3
- **Wait Between Tries:** 1000ms (exponential backoff)

### 3. Alertas Inteligentes
Solo notifica si errores > 5%:

```javascript
const errorRate = invalid_count / (valid_count + invalid_count);
if (errorRate > 0.05) {
  // Send alert
}
```

## Ejemplo Completo de Export/Import

El workflow completo está disponible en:
`recursos/guias-n8n/workflow-1-etl-basico.json`

Importa en n8n:
1. Workflow → Import from File
2. Selecciona archivo JSON
3. Ajusta credenciales
4. Ejecuta

## Checklist de Deployment

Antes de activar el workflow en producción:
- [ ] Credenciales configuradas correctamente
- [ ] Schedule configurado apropiadamente
- [ ] Error handling implementado
- [ ] Notificaciones funcionando
- [ ] Logging activado
- [ ] Probado con datos reales
- [ ] Documentación actualizada

## Troubleshooting

**Problema:** Workflow falla en extracción de Google Sheets
- Verifica credenciales OAuth
- Confirma que Sheet ID es correcto
- Revisa permisos de la cuenta

**Problema:** Datos no se limpian correctamente
- Revisa logs del Code node
- Verifica formato de datos de entrada
- Añade console.log para debugging

**Problema:** Notificaciones no llegan
- Verifica token de Slack
- Confirma que bot está en el canal
- Revisa formato del mensaje

## Recursos Adicionales

- [n8n Documentation - Google Sheets](https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.googlesheets/)
- [n8n Documentation - Code Node](https://docs.n8n.io/code-examples/)
- [Pandas Documentation](https://pandas.pydata.org/docs/) (para Code node Python)

---

**Siguiente:** [Guía 2: Orquestación de Entrenamiento/Inferencia](./guia-2-orquestacion-ml.md)
