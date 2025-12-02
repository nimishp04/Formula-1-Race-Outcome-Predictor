"""
F1 Race Predictor - Data Processing (Fixed Path Handling)
Processes raw F1 CSV data into usable format
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

class F1DataProcessor:
    def __init__(self):
        self.races = None
        self.results = None
        self.drivers = None
        self.constructors = None
        self.qualifying = None
        self.circuits = None
        
        # Determine project root directory
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
    def load_data(self, data_dir=None):
        """Load all CSV files"""
        print("Loading data files...")
        
        # Try multiple possible locations
        if data_dir is None:
            possible_paths = [
                self.project_root / 'data_raw',
                self.script_dir / 'data_raw',
                Path('data_raw'),
                Path('../data_raw'),
                Path('.')  # Current directory
            ]
            
            # Find which path exists
            data_dir = None
            for path in possible_paths:
                if (path / 'races.csv').exists():
                    data_dir = path
                    print(f"✓ Found data in: {path.absolute()}")
                    break
            
            if data_dir is None:
                print("\n❌ ERROR: CSV files not found!")
                print("\nPlease do ONE of the following:")
                print("\n1. Download F1 dataset from Kaggle:")
                print("   https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020")
                print("\n2. Create 'data_raw' folder in project root:")
                print(f"   {self.project_root / 'data_raw'}")
                print("\n3. Place these CSV files in data_raw folder:")
                print("   - races.csv")
                print("   - results.csv")
                print("   - drivers.csv")
                print("   - constructors.csv")
                print("   - qualifying.csv")
                print("   - circuits.csv")
                print("\nThen run this script again.")
                raise FileNotFoundError("CSV files not found in any expected location")
        
        # Load CSV files
        try:
            self.races = pd.read_csv(data_dir / 'races.csv')
            self.results = pd.read_csv(data_dir / 'results.csv')
            self.drivers = pd.read_csv(data_dir / 'drivers.csv')
            self.constructors = pd.read_csv(data_dir / 'constructors.csv')
            self.qualifying = pd.read_csv(data_dir / 'qualifying.csv')
            self.circuits = pd.read_csv(data_dir / 'circuits.csv')
            
            print(f"✓ Loaded {len(self.races)} races")
            print(f"✓ Loaded {len(self.results)} results")
            print(f"✓ Loaded {len(self.drivers)} drivers")
            print(f"✓ Loaded {len(self.constructors)} constructors")
            print(f"✓ Loaded {len(self.qualifying)} qualifying sessions")
            print(f"✓ Loaded {len(self.circuits)} circuits")
            
        except Exception as e:
            print(f"\n❌ ERROR loading CSV files: {e}")
            print(f"\nMake sure all CSV files exist in: {data_dir}")
            raise
        
    def create_driver_database(self):
        """Create driver JSON database"""
        print("\nCreating drivers database...")
        drivers_list = []
        
        for _, driver in self.drivers.iterrows():
            # Calculate career statistics
            driver_results = self.results[self.results['driverId'] == driver['driverId']]
            
            wins = len(driver_results[driver_results['positionOrder'] == 1])
            podiums = len(driver_results[driver_results['positionOrder'] <= 3])
            races = len(driver_results)
            
            driver_data = {
                'id': int(driver['driverId']),
                'ref': driver['driverRef'],
                'number': str(driver['number']) if pd.notna(driver['number']) else None,
                'code': driver['code'] if pd.notna(driver['code']) else None,
                'forename': driver['forename'],
                'surname': driver['surname'],
                'dob': driver['dob'],
                'nationality': driver['nationality'],
                'stats': {
                    'races': int(races),
                    'wins': int(wins),
                    'podiums': int(podiums),
                    'winRate': round(wins / races * 100, 2) if races > 0 else 0,
                    'podiumRate': round(podiums / races * 100, 2) if races > 0 else 0
                }
            }
            
            drivers_list.append(driver_data)
        
        # Ensure data directory exists
        data_dir = self.project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        output_file = data_dir / 'drivers.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(drivers_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created drivers database: {output_file}")
        print(f"  {len(drivers_list)} drivers")
        
    def create_constructor_database(self):
        """Create constructor JSON database"""
        print("\nCreating constructors database...")
        constructors_list = []
        
        for _, constructor in self.constructors.iterrows():
            # Calculate team statistics
            constructor_results = self.results[self.results['constructorId'] == constructor['constructorId']]
            
            wins = len(constructor_results[constructor_results['positionOrder'] == 1])
            podiums = len(constructor_results[constructor_results['positionOrder'] <= 3])
            races = len(constructor_results)
            
            constructor_data = {
                'id': int(constructor['constructorId']),
                'ref': constructor['constructorRef'],
                'name': constructor['name'],
                'nationality': constructor['nationality'],
                'stats': {
                    'races': int(races),
                    'wins': int(wins),
                    'podiums': int(podiums),
                    'winRate': round(wins / races * 100, 2) if races > 0 else 0,
                    'podiumRate': round(podiums / races * 100, 2) if races > 0 else 0
                }
            }
            
            constructors_list.append(constructor_data)
        
        # Save to JSON
        data_dir = self.project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        
        output_file = data_dir / 'constructors.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(constructors_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created constructors database: {output_file}")
        print(f"  {len(constructors_list)} constructors")
        
    def create_circuit_database(self):
        """Create circuit JSON database"""
        print("\nCreating circuits database...")
        circuits_list = []
        
        for _, circuit in self.circuits.iterrows():
            # Calculate circuit statistics
            circuit_races = self.races[self.races['circuitId'] == circuit['circuitId']]
            
            circuit_data = {
                'id': int(circuit['circuitId']),
                'ref': circuit['circuitRef'],
                'name': circuit['name'],
                'location': circuit['location'],
                'country': circuit['country'],
                'lat': float(circuit['lat']),
                'lng': float(circuit['lng']),
                'alt': int(circuit['alt']) if pd.notna(circuit['alt']) else None,
                'stats': {
                    'totalRaces': len(circuit_races),
                    'firstRace': int(circuit_races['year'].min()) if len(circuit_races) > 0 else None,
                    'lastRace': int(circuit_races['year'].max()) if len(circuit_races) > 0 else None
                }
            }
            
            circuits_list.append(circuit_data)
        
        # Save to JSON
        data_dir = self.project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        
        output_file = data_dir / 'circuits.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(circuits_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created circuits database: {output_file}")
        print(f"  {len(circuits_list)} circuits")
        
    def create_historical_stats(self):
        """Create historical statistics"""
        print("\nCreating historical statistics...")
        
        stats = {
            'metadata': {
                'generatedAt': datetime.now().isoformat(),
                'totalRaces': len(self.races),
                'totalDrivers': len(self.drivers),
                'totalConstructors': len(self.constructors),
                'yearRange': {
                    'start': int(self.races['year'].min()),
                    'end': int(self.races['year'].max())
                }
            },
            'gridPositionAnalysis': self.analyze_grid_positions(),
            'topDrivers': self.get_top_drivers(),
            'topConstructors': self.get_top_constructors()
        }
        
        # Save to JSON
        data_dir = self.project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        
        output_file = data_dir / 'historical.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created historical statistics: {output_file}")
        
    def analyze_grid_positions(self):
        """Analyze win rates by grid position"""
        analysis = {}
        
        for grid in range(1, 21):
            grid_results = self.results[self.results['grid'] == grid]
            wins = len(grid_results[grid_results['positionOrder'] == 1])
            total = len(grid_results)
            
            if total > 0:
                analysis[f'P{grid}'] = {
                    'totalStarts': int(total),
                    'wins': int(wins),
                    'winRate': round(wins / total * 100, 2)
                }
        
        return analysis
        
    def get_top_drivers(self, n=10):
        """Get top N drivers by wins"""
        driver_wins = self.results[self.results['positionOrder'] == 1].groupby('driverId').size()
        top_drivers = driver_wins.nlargest(n)
        
        result = []
        for driver_id, wins in top_drivers.items():
            driver = self.drivers[self.drivers['driverId'] == driver_id].iloc[0]
            result.append({
                'name': f"{driver['forename']} {driver['surname']}",
                'wins': int(wins),
                'nationality': driver['nationality']
            })
        
        return result
        
    def get_top_constructors(self, n=10):
        """Get top N constructors by wins"""
        constructor_wins = self.results[self.results['positionOrder'] == 1].groupby('constructorId').size()
        top_constructors = constructor_wins.nlargest(n)
        
        result = []
        for constructor_id, wins in top_constructors.items():
            constructor = self.constructors[self.constructors['constructorId'] == constructor_id].iloc[0]
            result.append({
                'name': constructor['name'],
                'wins': int(wins),
                'nationality': constructor['nationality']
            })
        
        return result
    
    def process_all(self):
        """Process all data and create databases"""
        print("\n" + "="*50)
        print("F1 Data Processing")
        print("="*50 + "\n")
        
        try:
            self.load_data()
            
            print("\nCreating databases...")
            self.create_driver_database()
            self.create_constructor_database()
            self.create_circuit_database()
            self.create_historical_stats()
            
            print("\n" + "="*50)
            print("✅ SUCCESS! Data processing complete!")
            print("="*50)
            print(f"\nJSON files created in: {self.project_root / 'data'}")
            print("\nYou can now use the F1 Race Predictor!")
            
        except FileNotFoundError as e:
            print(f"\n{'='*50}")
            print("❌ SETUP REQUIRED")
            print("="*50)
            # Error message already printed in load_data()
            return False
        except Exception as e:
            print(f"\n{'='*50}")
            print("❌ ERROR")
            print("="*50)
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

if __name__ == '__main__':
    processor = F1DataProcessor()
    success = processor.process_all()
    
    if not success:
        print("\nPress Enter to exit...")
        input()