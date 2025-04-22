const ClassroomData = require("../models/ClassroomData");
const thingspeakService = require("./thingspeakService");

class DataInitializationService {
    /**
     * Check if there is data in the database for a specific classroom
     * @param {string} classroomId - The ID of the classroom
     * @param {Date} startDate - Start date for the data period
     * @param {Date} endDate - End date for the data period
     * @returns {Promise<boolean>} - True if data exists, false otherwise
     */
    async hasData(classroomId, startDate, endDate) {
        try {
            const count = await ClassroomData.countDocuments({
                classroomId,
                timestamp: { $gte: startDate, $lte: endDate },
            });
            return count > 0;
        } catch (error) {
            console.error(`Error checking for data existence for ${classroomId}:`, error.message);
            return false;
        }
    }

    /**
     * Check if there is data in the database for any classroom
     * @returns {Promise<boolean>} - True if data exists, false otherwise
     */
    async hasAnyData() {
        try {
            const count = await ClassroomData.countDocuments({});
            return count > 0;
        } catch (error) {
            console.error("Error checking for any data existence:", error.message);
            return false;
        }
    }

    /**
     * Initialize data by fetching from ThingSpeak for a specific classroom
     * @param {string} classroomId - The ID of the classroom
     * @param {Date} startDate - Start date for the data period (not used directly but useful for future implementation)
     * @param {Date} endDate - End date for the data period (not used directly but useful for future implementation)
     * @returns {Promise<boolean>} - True if initialization succeeded, false otherwise
     */
    async initializeDataForClassroom(classroomId, startDate, endDate) {
        try {
            console.log(`Initializing data for ${classroomId} from ThingSpeak...`);

            // Fetch a substantial amount of historical data (up to 8000 points)
            // ThingSpeak may limit this based on your account type
            const feeds = await thingspeakService.fetchData(classroomId, 8000);

            if (feeds.length === 0) {
                console.warn(`No data available from ThingSpeak for ${classroomId}`);
                return false;
            }

            await thingspeakService.processAndSaveData(classroomId, feeds);

            console.log(
                `Data initialization complete for ${classroomId}. Fetched ${feeds.length} records.`
            );
            return true;
        } catch (error) {
            console.error(`Error initializing data for ${classroomId}:`, error.message);
            return false;
        }
    }

    /**
     * Initialize data by fetching from ThingSpeak for all configured classrooms
     * @returns {Promise<boolean>} - True if initialization succeeded for at least one classroom, false otherwise
     */
    async initializeAllData() {
        try {
            console.log("Initializing data for all classrooms from ThingSpeak...");

            // Check if we already have data to avoid unnecessary API calls
            const hasData = await this.hasAnyData();
            if (hasData) {
                console.log("Data already exists in the database. Skipping initialization.");
                return true;
            }

            // Get all classroom IDs from the ThingSpeak service
            const classroomIds = Object.keys(thingspeakService.readApiKeys);
            let success = false;

            for (const classroomId of classroomIds) {
                // Avoid ThingSpeak rate limits by waiting between requests
                if (classroomId !== classroomIds[0]) {
                    console.log("Waiting 15 seconds to avoid ThingSpeak rate limits...");
                    await new Promise((resolve) => setTimeout(resolve, 15000));
                }

                const result = await this.initializeDataForClassroom(classroomId);
                success = success || result;
            }

            return success;
        } catch (error) {
            console.error("Error initializing data for all classrooms:", error.message);
            return false;
        }
    }

    /**
     * Ensure data exists for the given period and classroom, fetching from ThingSpeak if needed
     * @param {string} classroomId - The ID of the classroom or 'all' for all classrooms
     * @param {Date} startDate - Start date for the data period
     * @param {Date} endDate - End date for the data period
     * @returns {Promise<boolean>} - True if data exists or was successfully initialized
     */
    async ensureDataExists(classroomId, startDate, endDate) {
        if (classroomId === "all") {
            // Check all classrooms
            const classroomIds = Object.keys(thingspeakService.readApiKeys);
            let hasDataForAll = true;

            for (const id of classroomIds) {
                const hasDataForClassroom = await this.hasData(id, startDate, endDate);
                if (!hasDataForClassroom) {
                    hasDataForAll = false;
                    console.log(
                        `No data found for ${id} between ${startDate} and ${endDate}. Initializing...`
                    );
                    await this.initializeDataForClassroom(id, startDate, endDate);
                }
            }

            return hasDataForAll;
        } else {
            // Check specific classroom
            const hasDataForClassroom = await this.hasData(classroomId, startDate, endDate);
            if (!hasDataForClassroom) {
                console.log(
                    `No data found for ${classroomId} between ${startDate} and ${endDate}. Initializing...`
                );
                return await this.initializeDataForClassroom(classroomId, startDate, endDate);
            }
            return true;
        }
    }
}

module.exports = new DataInitializationService();
