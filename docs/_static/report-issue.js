/**
 * Make the "Report Issue" footer link page-specific
 *
 * This script finds the Report Issue link and updates it to include
 * the current page URL and title as query parameters for GitHub Issues.
 */
document.addEventListener('DOMContentLoaded', function() {
    // Find the Report Issue link in the footer using aria-label
    const reportLink = document.querySelector('a[aria-label="Report Issue"]');

    if (reportLink) {
        // Get current page info
        const pageUrl = window.location.href;
        const pageTitle = document.title.split(' — ')[0] || document.title;

        // Build the issue URL with pre-filled information
        const baseUrl = 'https://github.com/underworldcode/underworld3/issues/new';
        const params = new URLSearchParams({
            'template': 'docs-issue.md',
            'labels': 'documentation',
            'title': '[Docs] ' + pageTitle,
            'body': '## Page URL\n' + pageUrl + '\n\n## Description\n<!-- Describe the problem or suggested improvement -->\n\n'
        });

        reportLink.href = baseUrl + '?' + params.toString();
    }
});
