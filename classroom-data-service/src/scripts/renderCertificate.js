require("dotenv").config();
const fs = require("fs");
const path = require("path");
const Handlebars = require("handlebars");

// Define the directory paths
const templatePath = path.join(__dirname, "../templates/certificate.html");
const certificateDir = path.join(__dirname, "../../certificates");
const outputDir = path.join(__dirname, "../../certificates/html");

// Create output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

/**
 * Format a date for display
 * @param {Date|string} date - The date to format
 * @returns {string} - Formatted date string
 */
function formatDate(date) {
    const d = new Date(date);
    return d.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
    });
}

/**
 * Render a certificate JSON to HTML
 * @param {string} certificatePath - Path to the JSON certificate file
 */
async function renderCertificate(certificatePath) {
    try {
        // Read the template and certificate data
        const templateSource = fs.readFileSync(templatePath, "utf8");
        const certificateData = JSON.parse(fs.readFileSync(certificatePath, "utf8"));

        // Prepare the template
        const template = Handlebars.compile(templateSource);

        // Add helper functions for the template
        Handlebars.registerHelper("toLowerCase", function (str) {
            return str.toLowerCase();
        });

        Handlebars.registerHelper("toFixed", function (number, decimals) {
            if (isNaN(parseFloat(number)) || !isFinite(number)) {
                return number;
            }
            return parseFloat(number).toFixed(decimals);
        });

        Handlebars.registerHelper("subtract", function (a, b) {
            return a - b;
        });

        // Format dates
        if (certificateData.period) {
            certificateData.period.startDate = formatDate(certificateData.period.startDate);
            certificateData.period.endDate = formatDate(certificateData.period.endDate);
        }

        if (certificateData.generatedAt) {
            certificateData.generatedAt = formatDate(certificateData.generatedAt);
        }

        // Render the template with the certificate data
        const html = template(certificateData);

        // Generate output filename
        const basename = path.basename(certificatePath, ".json");
        const outputPath = path.join(outputDir, `${basename}.html`);

        // Write the rendered HTML to file
        fs.writeFileSync(outputPath, html);

        console.log(`Certificate rendered to ${outputPath}`);
        return outputPath;
    } catch (error) {
        console.error(`Error rendering certificate: ${error.message}`);
        throw error;
    }
}

/**
 * Render all certificates in the certificate directory
 */
async function renderAllCertificates() {
    try {
        // Get all JSON files in the certificate directory
        const files = fs.readdirSync(certificateDir).filter((file) => file.endsWith(".json"));

        if (files.length === 0) {
            console.log("No certificate JSON files found");
            return;
        }

        console.log(`Found ${files.length} certificate files to render`);

        // Render each certificate
        for (const file of files) {
            const filePath = path.join(certificateDir, file);
            await renderCertificate(filePath);
        }

        console.log("All certificates rendered successfully");
    } catch (error) {
        console.error(`Error rendering certificates: ${error.message}`);
    }
}

// Main function to run the script
async function main() {
    try {
        // Parse command line arguments
        const args = process.argv.slice(2);
        const certificatePath = args[0]; // Optional specific certificate to render

        if (certificatePath) {
            // Render specific certificate
            const fullPath = path.resolve(certificatePath);
            if (!fs.existsSync(fullPath)) {
                throw new Error(`Certificate file not found: ${fullPath}`);
            }
            await renderCertificate(fullPath);
        } else {
            // Render all certificates
            await renderAllCertificates();
        }

        console.log("Certificate rendering complete");
        process.exit(0);
    } catch (error) {
        console.error("Error:", error.message);
        process.exit(1);
    }
}

// Run the script
main();
