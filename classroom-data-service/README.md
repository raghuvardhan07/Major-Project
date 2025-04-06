# Classroom Data Service

A Node.js service that periodically fetches classroom environmental data from ThingSpeak and stores it in MongoDB Atlas for historical analysis and reporting.

## Features

- Fetches data from ThingSpeak API for multiple classrooms
- Stores data in MongoDB Atlas for historical records
- Runs on a scheduled basis (every 15 minutes)
- Handles rate limiting and error recovery
- Avoids duplicate data entries

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

### Development Mode
```
npm run dev
```

### Production Mode
```
npm start
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

## Extending

To add more classrooms:

1. Add new classroom details to your `.env` file
2. The service will automatically detect and fetch data for all configured classrooms

## Troubleshooting

- **Connection Issues**: Check your MongoDB URI and network connectivity
- **No Data**: Verify your ThingSpeak API keys and channel IDs
- **Rate Limiting**: Free ThingSpeak accounts have rate limits - the service respects these with appropriate delays 