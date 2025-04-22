const certificateService = require("../services/certificateService");

/**
 * Generate a certificate for a classroom
 * @param {string} classroomId - The ID of the classroom
 * @param {Date} startDate - Start date for the period
 * @param {Date} endDate - End date for the period
 * @returns {Object} - Certificate data
 */
async function generateCertificate(classroomId, startDate, endDate) {
    try {
        // Validate inputs
        if (!classroomId || !["classroom1", "classroom2"].includes(classroomId)) {
            throw new Error("Invalid classroom ID");
        }

        // Parse dates if strings were provided
        const parsedStartDate = startDate instanceof Date ? startDate : new Date(startDate);
        const parsedEndDate = endDate instanceof Date ? endDate : new Date(endDate);

        // Validate dates
        if (isNaN(parsedStartDate.getTime()) || isNaN(parsedEndDate.getTime())) {
            throw new Error("Invalid date format");
        }

        // Generate the certificate
        const certificate = await certificateService.generateCertificate(
            classroomId,
            parsedStartDate,
            parsedEndDate
        );

        return certificate;
    } catch (error) {
        console.error("Error in certificate controller:", error.message);
        throw error;
    }
}

/**
 * Generate certificates for all classrooms for the same period
 * @param {Date} startDate - Start date for the period
 * @param {Date} endDate - End date for the period
 * @returns {Object} - Object with certificates for each classroom
 */
async function generateAllCertificates(startDate, endDate) {
    try {
        // Parse dates if strings were provided
        const parsedStartDate = startDate instanceof Date ? startDate : new Date(startDate);
        const parsedEndDate = endDate instanceof Date ? endDate : new Date(endDate);

        // Validate dates
        if (isNaN(parsedStartDate.getTime()) || isNaN(parsedEndDate.getTime())) {
            throw new Error("Invalid date format");
        }

        // Generate certificates for all classrooms
        const results = {};

        // Run the certificate generation in parallel
        const promises = ["classroom1", "classroom2"].map(async (classroomId) => {
            try {
                const certificate = await certificateService.generateCertificate(
                    classroomId,
                    parsedStartDate,
                    parsedEndDate
                );
                return { classroomId, certificate };
            } catch (error) {
                console.error(`Error generating certificate for ${classroomId}:`, error.message);
                return { classroomId, error: error.message };
            }
        });

        const certificateResults = await Promise.all(promises);

        // Convert array of results to object
        certificateResults.forEach((result) => {
            if (result.certificate) {
                results[result.classroomId] = result.certificate;
            } else {
                results[result.classroomId] = { error: result.error };
            }
        });

        return results;
    } catch (error) {
        console.error("Error in certificate controller:", error.message);
        throw error;
    }
}

module.exports = {
    generateCertificate,
    generateAllCertificates,
};
