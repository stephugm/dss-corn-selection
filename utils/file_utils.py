"""
Utility functions for handling CSV file operations.
"""
import csv
import os
from typing import List, Dict, Any, Optional

class CSVHandler:
    def __init__(self, data_dir: str = 'data'):
        """Initialize CSV handler with data directory path."""
        self.data_dir = data_dir
        self._ensure_data_dir_exists()
    
    def _ensure_data_dir_exists(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_file_path(self, filename: str) -> str:
        """Get full path for a CSV file."""
        return os.path.join(self.data_dir, filename)
    
    def read_all(self, filename: str) -> List[List[str]]:
        """Read all rows from a CSV file."""
        file_path = self._get_file_path(filename)
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            return list(reader)
    
    def read_by_id(self, filename: str, project_id: str) -> List[List[str]]:
        """Read rows from a CSV file filtered by project_id."""
        rows = self.read_all(filename)
        return [row for row in rows if row and row[0] == project_id]
    
    def append_row(self, filename: str, row: List[Any]):
        """Append a single row to a CSV file."""
        file_path = self._get_file_path(filename)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def append_rows(self, filename: str, rows: List[List[Any]]):
        """Append multiple rows to a CSV file."""
        file_path = self._get_file_path(filename)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def update_by_id(self, filename: str, project_id: str, new_data: List[List[Any]]):
        """Update rows in a CSV file for a specific project_id."""
        rows = self.read_all(filename)
        # Remove existing rows for this project
        rows = [row for row in rows if row and row[0] != project_id]
        # Add new rows
        rows.extend(new_data)
        
        file_path = self._get_file_path(filename)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def delete_by_id(self, filename: str, project_id: str):
        """Delete all rows for a specific project_id from a CSV file."""
        rows = self.read_all(filename)
        rows = [row for row in rows if row and row[0] != project_id]
        
        file_path = self._get_file_path(filename)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def delete_project(self, project_id: str):
        """Delete all data related to a project from all CSV files."""
        files = ['projects.csv', 'criteria.csv', 'weights.csv', 
                'alternatives.csv', 'scores.csv', 'results.csv']
        for file in files:
            self.delete_by_id(file, project_id)
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project data by ID."""
        rows = self.read_by_id('projects.csv', project_id)
        if not rows:
            return None
        
        row = rows[0]
        return {
            'id': row[0],
            'title': row[1],
            'description': row[2],
            'method': row[3]
        }
        
    def write_all(self, filename: str, rows: List[List[Any]]):
        """Write all rows to a CSV file, overwriting if it exists."""
        file_path = self._get_file_path(filename)
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
