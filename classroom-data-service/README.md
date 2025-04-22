# Classroom Data Service

A Node.js service that periodically fetches classroom environmental data from ThingSpeak and stores it in MongoDB Atlas for historical analysis and reporting.

## Features

- Fetches data from ThingSpeak API for multiple classrooms
- Stores data in MongoDB Atlas for historical records
- Runs on a scheduled basis (every 15 minutes)
- Handles rate limiting and error recovery
- Avoids duplicate data entries
- Generates classroom environmental quality certificates based on historical data
- Web interface for generating and viewing certificates
- Automatic data initialization from ThingSpeak when needed

## Prerequisites

- Node.js (v14 or higher)
- MongoDB Atlas account
- ThingSpeak account with channel(s) set up

## Setup

1. Clone this repository
2. Install dependencies
   ```
   npm install
   ```
3. Copy the environment file and update it with your credentials
   ```
   cp .env.example .env
   ```
4. Update `.env` with your:
   - MongoDB Atlas connection string
   - ThingSpeak Channel IDs
   - ThingSpeak Read API Keys

## MongoDB Atlas Setup

1. Create a MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster
3. Create a database user with read/write permissions
4. Add your IP address to the IP access list
5. Get your connection string from the Connect dialog
6. Replace `<username>`, `<password>`, `<cluster>`, and `<dbname>` in the MONGODB_URI in your `.env` file

## ThingSpeak Setup

1. Sign in to your ThingSpeak account
2. Note your Channel IDs for each classroom
3. Go to the API Keys tab and copy your Read API Keys
4. Add these values to your `.env` file

## Running the Service

### ThingSpeak Data Collection Service
```
npm start     # Production mode
npm run dev   # Development mode with auto-reload
```

### Certificate Generation Web Server
```
npm run server      # Production mode
npm run server:dev  # Development mode with auto-reload
```

Once the server is running, access the certificate generator at:
```
http://localhost:3000
```

## Automatic Data Initialization

The system is designed to work smoothly even when starting with an empty database:

1. When the server starts, it checks if there's any data in the database
2. If no data exists, it automatically fetches historical data from ThingSpeak
3. During certificate generation, if data for the requested period is missing, it's fetched on-demand
4. The UI displays a loading indicator with additional context when initialization is in progress

This ensures that users can generate certificates even on the first run, without any manual data import steps.

## Certificate Generation

The service can generate classroom environmental quality certificates based on historical data collected from ThingSpeak. Certificates include:

- Overall environmental quality score (0-100) based on a sigmoid function of KPIv values
- Letter grade (A+, A, B, C, D, F) based on the score
- Average metrics (temperature, humidity, CO2, occupancy, KPIv)
- Min/max values for each metric
- Customized recommendations based on the data

### Using the Web Interface

1. Start the certificate web server: `npm run server`
2. Open your browser and navigate to: `http://localhost:3000`
3. Select a classroom (or all classrooms)
4. Choose the date range
5. Click "Generate Certificate"
6. View, print, or download the generated certificate

### Command Line Certificate Generation

Generate a certificate for a specific classroom and date range:

```
npm run generate-cert classroom1 2023-01-01 2023-12-31
```

Generate certificates for all classrooms:

```
npm run generate-cert
```

The certificates will be saved as JSON files in the `certificates` directory.

### Rendering HTML Certificates from Command Line

Convert the JSON certificates to styled HTML documents:

```
npm run render-cert
```

This will render all certificate JSON files to HTML files in the `certificates/html` directory.

You can also render a specific certificate:

```
npm run render-cert ./certificates/classroom1_certificate_2023-01-01_to_2023-12-31.json
```

## Data Schema

The service stores the following data for each classroom reading:

- `classroomId`: Identifier for the classroom (classroom1 or classroom2)
- `timestamp`: Time when the reading was taken
- `temperature`: Temperature in degrees Celsius
- `humidity`: Humidity percentage
- `co2`: CO2 level in ppm
- `occupancy`: Number of people in the room
- `kpiv`: Ventilation KPI value
- `trend`: Trend prediction value
- `alertStatus`: Alert status code (0=normal, 1=warning, 2=critical)
- `modelVersion`: Version of the prediction model used
- `entryId`: Unique entry ID from ThingSpeak

## Certificate Scoring

The certificate scoring system uses a sigmoid function to convert KPIv values (typically 0-2) to a score between 0 and 100:

```javascript
// Lower KPIv is better, so we invert it: 2-kpiv
const invertedKpiv = 2 - normalizedKpiv;

// Sigmoid function: 100 * (1 / (1 + e^(-k * (x - midpoint))))
const score = 100 * (1 / (1 + Math.exp(-3 * (invertedKpiv - 1))));
```

The score is then mapped to a letter grade:
- A+: 95-100
- A: 90-94
- B: 80-89
- C: 70-79
- D: 60-69
- F: 0-59

## Extending

To add more classrooms:

1. Add new classroom details to your `.env` file
2. The service will automatically detect and fetch data for all configured classrooms

## Troubleshooting

- **Connection Issues**: Check your MongoDB URI and network connectivity
- **No Data**: Verify your ThingSpeak API keys and channel IDs
- **Rate Limiting**: Free ThingSpeak accounts have rate limits - the service respects these with appropriate delays
- **Empty Certificates**: If you receive an error about no data available, the date range you selected might not have any records. Try selecting a broader date range or check that your ThingSpeak channel has data for the selected period. 