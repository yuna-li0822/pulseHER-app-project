import React, { useState, useRef, useEffect } from 'react';
import './CameraPPG.css';

const CameraPPG = ({ duration = 30, onComplete }) => {
	const [isRecording, setIsRecording] = useState(false);
	const [status, setStatus] = useState('idle');
	const [ppgData, setPpgData] = useState([]);
	const [lastSample, setLastSample] = useState(null);
	const [snapshotUrl, setSnapshotUrl] = useState(null);
	const [bypassBrightness, setBypassBrightness] = useState(false);

	const videoRef = useRef(null);
	const canvasRef = useRef(null);
	const graphRef = useRef(null);
	const rafRef = useRef(null);
	const streamRef = useRef(null);
	const frameCountRef = useRef(0);

	const isRecordingRef = useRef(false);

	useEffect(() => {
		// Ensure the ref is in sync with the state
		isRecordingRef.current = isRecording;
	}, [isRecording]);

	useEffect(() => {
		return () => {
			stopAll();
		};
	}, []);

	const setStatusMsg = (s) => {
		try { setStatus(s); } catch (e) {}
	};

	async function requestPermission() {
		try {
			setStatusMsg('requesting camera permission...');
			const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
			streamRef.current = s;
			if (videoRef.current) {
				videoRef.current.srcObject = s;
				videoRef.current.onloadedmetadata = () => {
					setStatusMsg('camera ready, click start');
				};
			}
			setStatusMsg('camera permission granted');
			return s;
		} catch (err) {
			console.error('permission error', err);
			setStatusMsg('camera permission denied: ' + (err && err.name ? err.name : String(err)));
			throw err;
		}
	}

	async function attemptPlay() {
		try {
			if (!videoRef.current) return;
			await videoRef.current.play();
			setStatusMsg('video playing');
		} catch (e) {
			setStatusMsg('video play blocked - tap allowed');
		}
	}

	async function startRecording() {
		try {
			setStatusMsg('starting...');
			if (!streamRef.current) {
				try {
					await requestPermission();
				} catch (err) {
					return;
				}
			}

			if (videoRef.current && !videoRef.current.srcObject && streamRef.current) {
				videoRef.current.srcObject = streamRef.current;
			}
			
			videoRef.current.play().then(() => {
				setStatusMsg('video playback started');
				setIsRecording(true);
				frameCountRef.current = 0;
				setPpgData([]);
				
				// Give it a moment for playback to stabilize before starting the loop
				setTimeout(() => {
					startLoop();
				}, 300);

			}).catch(e => {
				setStatusMsg('playback failed, tap video to try');
				console.error("Playback failed", e);
			});

		} catch (err) {
			console.error('startRecording error', err);
			setStatusMsg('start error: ' + err.message);
		}
	}

	function stopAll() {
		setIsRecording(false);
		if (rafRef.current) {
			try { cancelAnimationFrame(rafRef.current); } catch (e) {}
			rafRef.current = null;
		}
		if (streamRef.current) {
			try {
				streamRef.current.getTracks().forEach(t => t.stop());
			} catch (e) {}
			streamRef.current = null;
		}
		if (videoRef.current) {
			try { videoRef.current.srcObject = null; } catch (e) {}
		}
		setStatusMsg('stopped');
	}

	function stopRecording() {
		stopAll();
		if (onComplete) onComplete(ppgData);
	}

	function extractAvgRed(ctx, w, h) {
		try {
			const cx = Math.floor(w / 2);
			const cy = Math.floor(h / 2);
			const sampleSize = Math.max(8, Math.floor(Math.min(50, Math.min(w, h) * 0.12)));
			const sx = Math.max(0, Math.min(w - 1, Math.floor(cx - sampleSize / 2)));
			const sy = Math.max(0, Math.min(h - 1, Math.floor(cy - sampleSize / 2)));
			const sw = Math.max(1, Math.min(sampleSize, w - sx));
			const sh = Math.max(1, Math.min(sampleSize, h - sy));
			const data = ctx.getImageData(sx, sy, sw, sh).data;
			let red = 0;
			for (let i = 0; i < data.length; i += 4) red += data[i];
			const avg = red / Math.max(1, data.length / 4);
			if (!bypassBrightness && (avg < 30 || avg > 245)) return null;
			return { avg, sx, sy, sw, sh };
		} catch (e) {
			return null;
		}
	}

	function drawWaveform(dataArr) {
		try {
			const c = graphRef.current;
			if (!c) return;
			const ctx = c.getContext('2d');
			const w = c.width;
			const h = c.height;
			ctx.clearRect(0, 0, w, h);
			if (!dataArr || dataArr.length === 0) return;
			const minV = Math.min(...dataArr);
			const maxV = Math.max(...dataArr);
			const range = Math.max(1e-6, maxV - minV);
			ctx.strokeStyle = '#00c853';
			ctx.lineWidth = 2;
			ctx.beginPath();
			const step = w / Math.max(1, dataArr.length - 1);
			for (let i = 0; i < dataArr.length; i++) {
				const v = (dataArr[i] - minV) / range;
				const y = h - (10 + v * (h - 20));
				const x = i * step;
				if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
			}
			ctx.stroke();
		} catch (e) {}
	}

	function startLoop() {
		const canvas = canvasRef.current;
		const video = videoRef.current;
		if (!canvas || !video) {
			setStatusMsg('error: video/canvas missing');
			return;
		}
		const ctx = canvas.getContext('2d');
		setStatusMsg('starting analysis loop...');

		const step = () => {
			try {
				if (!isRecordingRef.current) { // Use the ref here for immediate consistency
					setStatusMsg('loop stopped.');
					return;
				}
				if (video.readyState >= 2 && video.videoWidth > 0) {
					canvas.width = video.videoWidth;
					canvas.height = video.videoHeight;
					ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
					const sample = extractAvgRed(ctx, canvas.width, canvas.height);
					if (sample) {
						const scaled = sample.avg;
						frameCountRef.current++;
						setPpgData(prev => {
							const next = prev.slice(-599);
							next.push(scaled);
							drawWaveform(next);
							return next;
						});
						setLastSample(sample.avg);
						setStatusMsg('recording... frames: ' + frameCountRef.current);
						ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2; ctx.strokeRect(sample.sx, sample.sy, sample.sw, sample.sh); ctx.restore();
					} else {
						setStatusMsg('cover camera lens completely');
					}
				} else {
					setStatusMsg('waiting for video frames...');
				}
			} catch (e) {
				console.warn('loop error', e);
				setStatusMsg('analysis loop error');
			}
			rafRef.current = requestAnimationFrame(step);
		};
		rafRef.current = requestAnimationFrame(step);
	}

	function takeSnapshot() {
		try {
			const c = canvasRef.current;
			if (!c) return;
			const url = c.toDataURL('image/png');
			setSnapshotUrl(url);
			setStatusMsg('snapshot taken');
		} catch (e) { setStatusMsg('snapshot failed'); }
	}

	return (
		<div className="camera-ppg">
			<div style={{ marginBottom: 8 }}><strong>Status:</strong> {status}</div>

			<div className="video-container">
				<video ref={videoRef} className="ppg-video" playsInline muted onClick={attemptPlay} style={{ display: 'none' }} />
				<canvas ref={canvasRef} className="ppg-canvas" style={{ display: 'block' }} />
			</div>

			<div style={{ marginTop: 12 }}>
				<button onClick={() => requestPermission()} style={{ marginRight: 8 }}>Request Camera Permission</button>
				{!isRecording ? (
					<button onClick={() => startRecording()} style={{ marginRight: 8 }}>Start Measurement</button>
				) : (
					<button onClick={() => stopRecording()} style={{ marginRight: 8 }}>Stop</button>
				)}
				<button onClick={takeSnapshot} style={{ marginRight: 8 }}>Snapshot</button>
				<label style={{ marginLeft: 8 }}><input type="checkbox" checked={bypassBrightness} onChange={(e) => setBypassBrightness(e.target.checked)} /> Bypass brightness filter</label>
			</div>

			<div style={{ marginTop: 12 }}>
				<h4>Live Metrics</h4>
				<div>Last sample: {lastSample ? lastSample.toFixed(1) : '--'}</div>
				<canvas ref={graphRef} width={600} height={140} style={{ width: '100%', height: 140, background: '#fff', borderRadius: 8, border: '1px solid rgba(0,0,0,0.06)' }} />
			</div>

			{snapshotUrl && (
				<div style={{ marginTop: 12 }}>
					<h4>Snapshot</h4>
					<img src={snapshotUrl} alt="snapshot" style={{ maxWidth: '100%' }} />
				</div>
			)}
		</div>
	);
};

export default CameraPPG;

