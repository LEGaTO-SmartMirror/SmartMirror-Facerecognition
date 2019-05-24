    
/**
 * @file smartmirror-facerecognition.js
 *
 * @author nkucza
 * @license MIT
 *
 * @see  https://github.com/NKucza/smartmirror-facerecognition
 */


Module.register('SmartMirror-Facerecognition',{

	defaults: {

	},

	// Override socket notification handler.
	socketNotificationReceived: function(notification, payload) {
		if ( notification == 'recognised_identities') {
			//console.log("[" + this.name + "] detected persons: " + payload);
			this.sendNotification("FACE_REC_IDS" , payload);
		}else if ( notification == 'recognition') {
			this.sendNotification("FACE_REC_DETECTIONS" , payload);
		}else if ( notification == 'DETECTED_FACES') {
			this.sendNotification("DETECTED_FACES" , payload);
		}else if (notification === 'FACE_DET_FPS') {
			this.sendNotification('FACE_DET_FPS', payload);
		};
	},

	notificationReceived: function(notification, payload, sender) {
		if(notification === 'smartmirror-facerecognitionSetFPS') {
			this.sendSocketNotification('FaceRecognition_SetFPS', payload);
        };
	},

	start: function() {
		this.time_of_last_greeting_personal = [];
		this.time_of_last_greeting = 0;
		this.last_rec_user = [];
		this.current_user = null;
		this.sendSocketNotification('FACEDETECTION_CONFIG', this.config);
		Log.info('Starting module: ' + this.name);
	}

});
