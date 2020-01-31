'use strict';
const NodeHelper = require('node_helper');

const {PythonShell} = require("python-shell");
const { exec } = require('child_process');
var pythonStarted = false;

module.exports = NodeHelper.create({

 	python_start: function () {
		const self = this;
    		self.pyshell = new PythonShell('modules/' + this.name + '/facerecognition/facerecognition_tracked.py', { pythonPath: 'python3', args: [JSON.stringify(this.config)]});


    		self.pyshell.on('message', function (message) {
			try {
				var parsed_message = JSON.parse(message)
				//console.log("[MSG " + self.name + "] " + parsed_message);
      				if (parsed_message.hasOwnProperty('status')){
      					console.log("[" + self.name + "] " + parsed_message.status);
      				}else if (parsed_message.hasOwnProperty('DETECTED_FACES')){
					self.sendSocketNotification('DETECTED_FACES', parsed_message)
					//console.log("[" + self.name + "] DETECTED_FACES: " + JSON.stringify(parsed_message));
				}else if (parsed_message.hasOwnProperty('FACE_DET_FPS')){
					self.sendSocketNotification('FACE_DET_FPS', parsed_message.FACE_DET_FPS);
					//console.log("[" + self.name + "] face detection fps: " + JSON.stringify(parsed_message));
				}	
			}
			catch(err) {
				console.log("[" + self.name + "] a non json message received");
			}
    		});
			exec(`renice -n 20 -p ${self.pyshell.childProcess.pid}`,(error,stdout,stderr) => {
				if (error) {
					console.error(`exec error: ${error}`);
  				}
			});
	
  	},

  // Subclass socketNotificationReceived received.
  socketNotificationReceived: function(notification, payload) {
 		const self = this;
 		if(notification === 'FaceRecognition_SetFPS') {
			if(pythonStarted) {
 				var data = {"FPS": payload}
                self.pyshell.send(JSON.stringify(data));
            }
        }else if(notification === 'FACEDETECTION_CONFIG') {
      		this.config = payload
      		if(!pythonStarted) {
        		this.python_start();
        		pythonStarted = true;
        	};
    	};
  },
	stop: function() {
		const self = this;
		self.pyshell.childProcess.kill('SIGINT');
		self.pyshell.end(function (err) {
           	if (err){
        		//throw err;
    		};
    		console.log('finished');
		});
		self.pyshell.childProcess.kill('SIGKILL');
	}

});
