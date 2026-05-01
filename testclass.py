import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import asyncio
import os
from collections import defaultdict
from visualization_service import VisualizationService
#import re
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nest_asyncio
import threading
#import matplotlib.ticker as ticker
#import matplotlib.patheffects as path_effects  # For text contrast
from tkinter import simpledialog

from logger import _log_unmatched_parts, _log_successful_processing, _build_success_response, _log_current_unmatches, _log_current_unmatches
from user_data_preparation import *
from utilities import *
from config import *
from validation import _validate_prediction_count, _validate_rating_values, ValidationError, _get_missing_predictions
from user_processing import *

