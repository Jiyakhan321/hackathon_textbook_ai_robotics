/**
 * Utility functions for handling text selection in the textbook
 */

export const getTextSelection = () => {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
        return selection.toString().trim();
    }
    return '';
};

export const getSelectedRange = () => {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
        return selection.getRangeAt(0);
    }
    return null;
};

export const highlightSelectedText = (element) => {
    // Create a highlight effect for selected text
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const highlightSpan = document.createElement('span');
        highlightSpan.style.backgroundColor = 'yellow';
        highlightSpan.style.padding = '1px 2px';
        
        // Wrap the selected text in the highlight span
        range.surroundContents(highlightSpan);
    }
};

export const copySelectedText = () => {
    const selectedText = getTextSelection();
    if (selectedText) {
        navigator.clipboard.writeText(selectedText)
            .then(() => console.log('Text copied to clipboard'))
            .catch(err => console.error('Failed to copy text: ', err));
    }
};

// Function to set up text selection functionality on a given element
export const setupTextSelection = (element, onTextSelected) => {
    let isSelecting = false;
    let startX, startY;

    const handleMouseDown = () => {
        isSelecting = true;
    };

    const handleMouseUp = () => {
        if (isSelecting) {
            const selectedText = getTextSelection();
            if (selectedText) {
                onTextSelected(selectedText);
            }
        }
        isSelecting = false;
    };

    // Add event listeners to the element
    element.addEventListener('mousedown', handleMouseDown);
    element.addEventListener('mouseup', handleMouseUp);
    
    // Also listen for selection changes
    document.addEventListener('selectionchange', () => {
        if (isSelecting) {
            const selectedText = getTextSelection();
            if (selectedText) {
                onTextSelected(selectedText);
            }
        }
    });

    // Return a function to remove event listeners
    return () => {
        element.removeEventListener('mousedown', handleMouseDown);
        element.removeEventListener('mouseup', handleMouseUp);
    };
};