# =============================================
# PulseHer Database Configuration
# =============================================
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

class PulseHerDatabase:
    def __init__(self):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate('firebase.json')
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def add_heart_data(self, user_id, heart_data):
        """Add heart monitoring data to Firestore"""
        try:
            doc_ref = self.db.collection('heart_data').document()
            data = {
                'user_id': user_id,
                'timestamp': datetime.now(),
                **heart_data
            }
            doc_ref.set(data)
            return {"success": True, "document_id": doc_ref.id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_heart_data(self, user_id, limit=10):
        """Get heart data for a user"""
        try:
            docs = self.db.collection('heart_data')\
                         .where('user_id', '==', user_id)\
                         .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                         .limit(limit)\
                         .stream()
            
            data = []
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data['id'] = doc.id
                data.append(doc_data)
            
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_user_profile(self, user_id, profile_data):
        """Add or update user profile"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            data = {
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                **profile_data
            }
            doc_ref.set(data, merge=True)
            return {"success": True, "user_id": user_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_profile(self, user_id):
        """Get user profile"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return {"success": True, "data": doc.to_dict()}
            else:
                return {"success": False, "error": "User not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == '__main__':
    db = PulseHerDatabase()
    
    # Example: Add sample heart data
    sample_data = {
        "bpm": 75,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "stress_level": 3,
        "activity": "Rest"
    }
    
    result = db.add_heart_data("user123", sample_data)
    print("Add data result:", result)
    
    # Example: Get heart data
    result = db.get_heart_data("user123")
    print("Get data result:", result)