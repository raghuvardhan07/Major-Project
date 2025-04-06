require("dotenv").config();
const cron = require("node-cron");
const connectDB = require("./config/database");
const thingspeakService = require("./services/thingspeakService");

// Connect to MongoDB
connectDB()
    .then(() => {
        console.log("Starting Classroom Data Service...");

        // Run immediately on startup
        runDataCollection();

        // Schedule the job to run every 15 minutes
        // This respects ThingSpeak's rate limiting for free accounts
        cron.schedule("*/15 * * * *", () => {
            console.log("Running scheduled data collection...");
            runDataCollection();
        });

        console.log("Classroom Data Service is running. Press Ctrl+C to exit.");
    })
    .catch((err) => {
        console.error("Failed to start service:", err);
        process.exit(1);
    });

// Function to run the data collection process
async function runDataCollection() {
    try {
        await thingspeakService.fetchAndSaveAllClassroomData();
    } catch (error) {
        console.error("Error in data collection process:", error);
    }
}

// Handle graceful shutdown
process.on("SIGINT", () => {
    console.log("Shutting down...");
    process.exit(0);
});

process.on("SIGTERM", () => {
    console.log("Shutting down...");
    process.exit(0);
});
