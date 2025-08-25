#!/usr/bin/env python3
"""
Test script for feature engineering module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_engineer import engineer_features, get_feature_summary

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    print("🧪 Testing Feature Engineering Module...")
    
    try:
        # Test feature engineering for Hungarian GP
        race_code = "2025_Hungarian_GP"
        
        print(f"🔄 Testing feature engineering for {race_code}...")
        features = engineer_features(race_code, 'data/processed/2025_Hungarian_GP_features.csv')
        
        # Get feature summary
        summary = get_feature_summary(features)
        
        # Display sample of features
        print(f"\n📋 Sample of engineered features:")
        print(features.head())
        
        print(f"\n✅ Feature engineering test completed successfully!")
        print(f"   - Generated {len(features)} driver records")
        print(f"   - Created {len(features.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    sys.exit(0 if success else 1)
