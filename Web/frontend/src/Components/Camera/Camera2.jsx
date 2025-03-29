import React, { useEffect, useRef, useState } from "react";
import { FaCameraRetro } from "react-icons/fa";
import { MdDriveFolderUpload, MdZoomInMap, MdZoomOutMap } from "react-icons/md";
export const Camera = ({setImageUpload}) => {
  const API_URL = "http://localhost:8000"
  // const API_URL = "https://d11f-34-143-212-184.ngrok-free.app"
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isSmall, setIsSmall] = useState(false);
  const [dataImage, setDataImage] = useState("");
  const [error, setError] = useState({
    "message": ""
  })
  const [imageBase64, setImageBase64] = useState("");

  useEffect(() => {
    const getWebcam = async () => {
      try { 
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        alert("Error accessing webcam");
        console.error("Error accessing webcam:", err);
      }
    };

    getWebcam();

    // Cleanup function
    return () => {
      const videoElement = videoRef.current; // 🔥 Lưu lại giá trị videoRef.current trước khi cleanup
      if (videoElement && videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (error.message === "Please wait a minute")
      return
    const timer = setTimeout(() => {
      setError({ "message": "" })
    }, 5000)
    return () => clearTimeout(timer);
  }, [error.message])

  const handleCapture = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext("2d");
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

      canvasRef.current.toBlob((blob) => {
        setDataImage(new File([blob], "imageCapture.png", { type: blob.type }))

      }, "image/png");
    }
  };

  const getProductSelected = () => {
    return JSON.parse(localStorage.getItem("product"));
  }

  const fetchApi = async () => {
    const pro = getProductSelected();
    if (!pro) {
      setError({ "message": "Please selected a product" })
      return;
    }
    try {
      setError({ "message": "Please wait a minute" })
      const response = await fetch(pro.image[0], {
        method: "GET",
        mode: "cors"
      });
      const blob = await response.blob();
      const formData = new FormData();
      formData.append("file2", dataImage);
      formData.append("file1", new File([blob], "heheh.png", { type: blob.type }));
      setDataImage("");
      const res = await fetch(`${API_URL}/Virtual-Try-On`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) return setError("Server not response")
      const data = await res.json();
      setError({ "message": "" })
      setImageBase64(data.image)
      setIsSmall(true)
    } catch (e) {
      console.log(e)
      setError({ "message": "Cannot connect server." })
    }
  }

  useEffect(() => {
    if (dataImage !== "" && dataImage) {
      setImageUpload(dataImage);
      fetchApi();
    }
  }, [dataImage]);


  const uploadFile = () => {
    let input = document.getElementById("uploadImage");

    input.addEventListener("change", (e) => {
      if (e.target.files[0] !== null) {
        setDataImage(e.target.files[0]);
        input.value = '';
      }
    })
    input.click();
  }

  const changeCamera = () => {

  }

  const handleZoom = () => {
    if(imageBase64 === ""){
      setError({ "message": "Please take a photo" })
      return;
    }
    setIsSmall(true)
  }
  return (
    <>
      <img className={`w-full h-full object-scale-down`} src={imageBase64 === "" ? "/imageDefault.png" : `data:image/png;base64,${imageBase64}`} alt="" />
      <div className={`group flex justify-center
      bg-black overflow-hidden rounded-2xl shadow-2xl 
      bottom-0 right-0 absolute
      transition-all duration-700 z-50 
    ${isSmall ? "cursor-pointer bottom-4 right-4 w-1/3 h-[120px]" : " h-full w-full"} `}>
        <video ref={videoRef} height="100%" autoPlay style={{ transform: 'scale(-1, 1)' }}></video>
        <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
        {!isSmall &&
          <>
            <div className="absolute bottom-12 right-1/2 transform translate-x-1/2 transition-all duration-200 inline-block text-nowrap ">
              <button onClick={() => handleZoom()} className=" p-3 rounded-full bg-white/40  cursor-pointer hover:bg-white/60"><MdZoomInMap className="text-2xl text-black" /></button>
              <button onClick={() => handleCapture()} className=" p-4 rounded-full bg-[#FF4141]/80  cursor-pointer hover:bg-[#ff4141]/60 mx-4"><FaCameraRetro className="text-2xl text-white" /></button>
              <button onClick={() => uploadFile()} className="p-3 rounded-full bg-white/40  cursor-pointer hover:bg-white/60"><MdDriveFolderUpload className="text-2xl text-black" /></button>
            </div>
            <input type="file" name="uploadImage" id="uploadImage" className="hidden" />
          </>
        }
        <h2 className={`absolute bottom-1/2 right-1/2 transform translate-x-1/2 translate-y-1/2 z-30 min-w-[100px] text-nowrap px-4 py-2 text-xl bg-gray-300 text-white text-center rounded-md bg-opacity-30 transition-all duration-200 ${error.message === "" ? "hidden" : "inline-block"}`}>{error.message}</h2>
        {isSmall &&
          <button className={`rounded-lg bg-[#EB423F]/90  hover:bg-[#EB423F] 
       text-white transition duration-200 absolute top-2 right-3 z-9999 p-1 invisible
       group-hover:visible `}
            onClick={() => setIsSmall(false)}
          ><MdZoomOutMap className={`text-stone-50`} /></button>}
      </div>
    </>

  );
};
