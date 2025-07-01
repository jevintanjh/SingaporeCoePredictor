
#!/usr/bin/env python3
"""
Comprehensive COE Model Testing Script
Generates detailed validation reports for all three prediction models
"""

import pandas as pd
import sys
import os
from utils.model_testing_validation import run_comprehensive_testing

def main():
    print("🧪 COE Model Comprehensive Testing Suite")
    print("=" * 50)
    
    # Load data
    try:
        data = pd.read_csv('data/COE_Extended_2002_2025.csv')
        data['date'] = pd.to_datetime(data['date'])
        print(f"✅ Loaded {len(data)} records from dataset")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run comprehensive testing
    print("\n🔬 Running comprehensive model testing...")
    tester = run_comprehensive_testing(data)
    
    if not tester.results:
        print("❌ No testing results generated")
        return
    
    print(f"✅ Testing completed for {len(tester.results)} models")
    
    # Generate detailed reports
    print("\n📊 COMPREHENSIVE MODEL TESTING RESULTS")
    print("=" * 60)
    
    for model_name in tester.results.keys():
        print(tester.generate_model_report(model_name))
        print("\n" + "="*80 + "\n")
    
    # Summary comparison
    print("📈 PERFORMANCE SUMMARY")
    print("-" * 30)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, results in tester.results.items():
        accuracy = results['overall']['direction_accuracy']
        r2 = results['overall']['r2']
        mape = results['overall']['mape']
        
        print(f"{model_name}:")
        print(f"  Direction Accuracy: {accuracy:.1%}")
        print(f"  R² Score: {r2:.3f}")
        print(f"  MAPE: {mape:.1f}%")
        print()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"🏆 BEST PERFORMING MODEL: {best_model} ({best_accuracy:.1%} accuracy)")
    
    # Feature importance summary
    print("\n🔍 FEATURE IMPORTANCE INSIGHTS")
    print("-" * 35)
    
    for model_name in tester.feature_importance.keys():
        print(f"\n{model_name} - Top 3 Features:")
        sorted_features = sorted(tester.feature_importance[model_name].items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i}. {feature}: {importance:.3f}")
    
    print(f"\n✅ Testing complete! Results available in testing output above.")

if __name__ == "__main__":
    main()
