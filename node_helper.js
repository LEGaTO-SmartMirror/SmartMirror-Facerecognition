'use strict';
const NodeHelper = require('node_helper');
const { spawn, exec } = require('child_process');

var cAppStarted = false

module.exports = NodeHelper.create({

	cApp_start: function () {
		const self = this;

		console.log("[" + self.name + "] starting face recognition")

		self.facerec = spawn('modules/' + self.name + '/FaceRecognition_CMake/build/FaceRecognition');

		self.facerec.stderr.on('data', (data) => {

		});

		self.facerec.stdout.on('data', (data) => {

			var data_chunks = `${data}`.split('\n');
			data_chunks.forEach( chunk => {

				if (chunk.length > 0) {
				try{
					var parsed_message = JSON.parse(chunk)
					if (parsed_message.hasOwnProperty('DETECTED_FACES')){
						//console.log("[" + self.name + "] Faces detected : " + JSON.stringify(parsed_message));
						self.sendSocketNotification('DETECTED_FACES', parsed_message);
					}else if (parsed_message.hasOwnProperty('FACE_DET_FPS')){
						//console.log("[" + self.name + "] " + JSON.stringify(parsed_message));
						self.sendSocketNotification('FACE_DET_FPS', parsed_message.FACE_DET_FPS);
					}else if (parsed_message.hasOwnProperty('STATUS')){
						console.log("[" + self.name + "] status received: " + JSON.stringify(parsed_message));
					}
				}
				catch(err) {
					if (err.message.includes("Unexpected token") && err.message.includes("in JSON")){
						console.log("[" + self.name + "] json parse error");
						console.log(chunk);
					} else if (err.message.includes("Unexpected end of JSON input")) {
						console.log("[" + self.name + "] Unexpected end of JSON input")
						console.log(chunk);
					} else {
						console.log(err.message)
					}
				}
				//console.log(chunk);
				}
			});
		});
	
		exec(`renice -n 5 -p ${self.facerec.pid}`,(error,stdout,stderr) => {
			if (error) {
				console.error(`exec error: ${error}`);
			}
		});

		self.facerec.on("exit", (code, signal) => {
			if (code !== 0){
				setTimeout(() => {self.cApp_start();}, 10)
			}
			console.log("facerec det: " + "code=" + code + " signal=" + signal);
		});
  	},

  // Subclass socketNotificationReceived received.
  socketNotificationReceived: function(notification, payload) {
 		const self = this;
 		if(notification === 'FaceRecognition_SetFPS') {
			if(cAppStarted) {
				self.facerec.stdin.write(payload.toString() + "\n");
				console.log("[" + self.name + "] changing to: " + payload.toString() + " FPS");
            }
        }else if(notification === 'FACEDETECTION_CONFIG') {
      		self.config = payload
      		if(!cAppStarted) {
        		self.cApp_start();
        		cAppStarted = true;
        	};
    	};
  }, 
	stop: function() {
		const self = this;
	/*	self.facerec.childProcess.kill('SIGINT');
		self.facerec.end(function (err) {
           	if (err){
        		//throw err;
    		};
    		console.log('finished');
		});
		self.facerec.childProcess.kill('SIGKILL'); */
	}

});
