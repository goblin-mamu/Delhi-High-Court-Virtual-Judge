import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def plot_case_similarity(similar_cases: List[Dict[str, Any]]):
    """
    Create a visualization of case similarity.
    
    Args:
        similar_cases: List of similar cases with similarity scores
        
    Returns:
        Figure object that can be displayed in Streamlit
    """
    # Extract case titles and similarity scores
    case_titles = []
    similarity_scores = []
    
    for case in similar_cases:
        # Truncate long titles
        title = case.get('title', f"Case {len(case_titles) + 1}")
        if len(title) > 30:
            title = title[:27] + "..."
        
        case_titles.append(title)
        similarity_scores.append(case.get('similarity_score', 0))
    
    # Sort by similarity score
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
    case_titles = [case_titles[i] for i in sorted_indices]
    similarity_scores = [similarity_scores[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up color gradient based on similarity scores
    colors = plt.cm.viridis(np.array(similarity_scores))
    
    y_pos = np.arange(len(case_titles))
    bars = ax.barh(y_pos, similarity_scores, color=colors)
    
    # Add labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(case_titles)
    ax.invert_yaxis()  # Cases with highest similarity at the top
    ax.set_xlabel('Similarity Score')
    ax.set_title('Case Similarity to Uploaded Document')
    
    # Add text labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center'
        )
    
    # Set x-axis limits
    ax.set_xlim(0, 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_judgment_distribution(similar_cases: List[Dict[str, Any]]):
    """
    Create a visualization of judgment distribution among similar cases.
    
    Args:
        similar_cases: List of similar cases
        
    Returns:
        Figure object that can be displayed in Streamlit
    """
    # Count judgments
    judgment_counts = {}
    
    for case in similar_cases:
        # In a real implementation, this would come from the case metadata
        # For now, we'll derive it from the case number
        case_number = case.get('case_number', '')
        
        # Extract numeric part from case number
        match = None
        if case_number:
            match = re.search(r'(\d+)/\d+', case_number)
        
        if match:
            num = int(match.group(1))
            
            # Determine judgment type
            if num % 5 == 0:
                judgment = "Allowed"
            elif num % 5 == 1:
                judgment = "Dismissed"
            elif num % 5 == 2:
                judgment = "Partly Allowed"
            elif num % 5 == 3:
                judgment = "Withdrawn"
            else:
                judgment = "Settled"
        else:
            judgment = "Unknown"
        
        if judgment in judgment_counts:
            judgment_counts[judgment] += 1
        else:
            judgment_counts[judgment] = 1
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(judgment_counts.keys())
    sizes = list(judgment_counts.values())
    
    # Color map
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Set title
    ax.set_title('Distribution of Judgments in Similar Cases')
    
    # Make text visible against various backgrounds
    for text in texts:
        text.set_color('black')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    return fig

import re
