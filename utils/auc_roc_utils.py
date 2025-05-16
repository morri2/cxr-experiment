from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray',
'darkred', 'darkgreen', 'darkblue', 'orange', 'pink', 'purple', 'brown',
'teal', 'lime', 'olive', 'navy', 'maroon', 'aqua', 'fuchsia', 'gold',
'silver', 'indigo', 'violet', 'coral', 'chocolate', 'crimson', 'darkcyan',
'darkorange', 'darkviolet', 'deeppink', 'forestgreen', 'hotpink', 'khaki',
'lavender', 'lightblue', 'lightgreen', 'mediumpurple', 'midnightblue', 'orchid', 'peru', 'plum', 'rosybrown', 'saddlebrown', 'salmon', 'sienna',
'skyblue', 'slateblue', 'slategray', 'springgreen', 'steelblue', 'tan',
'tomato', 'turquoise', 'wheat', 'yellowgreen']

def setup_roc_plot(title="ROC Curve"):
    """Create a new ROC plot with title and axis labels."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.grid(True)
    return fig, ax

def add_roc_curve(ax: Axes, fpr_tpr, label=None, color=None, linestyle="-", lw=2, alpha=0.8, auc_in_label=False):
    """Add a ROC curve to an existing plot."""
    fpr, tpr = fpr_tpr
    if auc_in_label:
        label = f"{label} (AUC={auc(fpr, tpr):.2f})"
    ax.plot(fpr, tpr, label=label, color=color, linestyle=linestyle, lw=lw, alpha=alpha)

def finalize_roc_plot(ax, loc="lower right", ncol=1):
    """Finalize and show a ROC plot."""
    ax.legend(loc=loc, ncol=ncol)
    # run plt.show() seperate
    