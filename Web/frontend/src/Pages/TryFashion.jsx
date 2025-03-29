import React, { useContext, useMemo, useRef, useState } from "react";
import { FaAngleDoubleLeft, FaChevronRight } from "react-icons/fa";
import { IoReload } from "react-icons/io5";
import { Camera } from "../Components/Camera/Camera2";
import { useEffect } from "react";
import { FiChevronsLeft } from "react-icons/fi";
import ReactPaginate from 'react-paginate';
import 'swiper/css';
import 'swiper/css/pagination';
import './CSS/slider.css';
import './CSS/PaginationMini.css'
import { ShopContext } from "../Context/ShopContext";
import { FaChevronLeft } from "react-icons/fa6";
import ItemTryFashionOn from "../Components/Item/ItemTryFashionOn";


export const TryFashion = () => {
  const itemsPerPage = 8;
  const [imageUpload, setImageUpload] = useState("");

  const { products2 } = useContext(ShopContext);
  const [currentData, setCurrentData] = useState([]);
  const [product, setProduct] = useState({});

  const [showTextCategory, setShowTextCategory] = useState(true);
  const [showInfor, setShowInfor] = useState(true);
  const [textSearch, setTextSearch] = useState("");
  const [category, setCategory] = useState("Full Product");

  const inputRef = useRef(null);
  const thisRef = useRef(null);

  const filteredData = useMemo(() => {
    if (products2 == null) return [];
    var filtered = products2.filter((item) => category === "Full Product" ? true : item.sex === category.toUpperCase());
    if (textSearch !== "") {
      filtered = filtered.filter(item => item.title && item.title.toLowerCase().includes(textSearch.toLowerCase()));
    }
    return filtered;
  }, [products2, category, textSearch]);

  useEffect(() => {
    setCurrentData(filteredData.slice(0, itemsPerPage));

  }, [filteredData]);

  useEffect(() => {
    if (currentData == null) return {};
    setProduct(currentData.at(0))
    localStorage.setItem("product", JSON.stringify(currentData.at(0)));
  }, [currentData]);


  const handleProductSelect = (idProductSelected) => {
    const pr_selected = currentData.filter(item => item._id === idProductSelected)[0];
    setProduct(pr_selected);
    localStorage.setItem("product", JSON.stringify(pr_selected));
  };

  const handlePageChange = (selectedPage) => {
    let startIndex = selectedPage.selected * itemsPerPage;
    const endIndex = Math.min(filteredData.length, startIndex + itemsPerPage);
    setCurrentData(filteredData.slice(startIndex, endIndex));
  };

  return (
    <div ref={thisRef} className="container mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 bg-gray-100">
      <section className="bg-white p-4 rounded-lg shadow-sm col-span-1 lg:col-span-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold mb-4">Product lists</h2>
          <ReactPaginate
            nextLabel={<FaChevronRight />}
            onPageChange={handlePageChange}
            pageCount={Math.ceil(filteredData.length / itemsPerPage)}
            previousLabel={<FaChevronLeft />}
            renderOnZeroPageCount={null}
            containerClassName="pagination"
            pageClassName="page-item"
            pageRangeDisplayed={1}
            marginPagesDisplayed={0}
            pageLinkClassName="page-link"
            previousClassName="page-item"
            previousLinkClassName="page-link"
            nextClassName="page-item"
            nextLinkClassName="page-link"
            activeClassName="active"
          />
        </div>
        <div className="flex items-center mb-4">
          <div
            onMouseOverCapture={() => { inputRef.current.blur() }}
            className={`group px-5 py-2 rounded-full border-black border-[1px]
           flex items-center justify-center cursor-pointer relative hover:bg-[#EB423F]/80
            hover:border-white transition-all duration-400 shadow-md inset-0 after:content-[''] 
            after:block after:w-full after:h-full after:absolute after:top-1/2 after:left-0
            ${showTextCategory ? "min-w-[180px]" : "min-w-[40px]"}
            `}>
            {showTextCategory && <p className="group-hover:text-white duration-200 transition-opacity">{category}</p>}
            <FiChevronsLeft className="ml-3 group-hover:-rotate-90 transition-all duration-200 text-lg group-hover:text-white" />

            <div className="absolute left-0 right-0 top-full p-2 rounded-lg mt-1 bg-white shadow-xl overflow-hidden hidden group-hover:block z-30">
              <ul>
                {["Full Product", "Female", "Male"].map((item, index) => (
                  <li key={index} className={`w-full px-4 py-1 hover:bg-gray-400 rounded-md mb-1 ${category === item ? "bg-gray-300" : ""}`} onClick={() => setCategory(item)}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
          <div className="relative ml-4 flex-1">
            <input ref={inputRef} className={`border rounded-full p-2 w-full outline-none px-4 transition-all 
            duration-400
            ${!showTextCategory ? 'border-black border-[1px]' : 'border-gray-300'}`}
              placeholder="Search.." type="text"
              onFocus={() => setShowTextCategory(!showTextCategory)}
              onBlur={() => setShowTextCategory(!showTextCategory)}
              onChange={Event => setTextSearch(Event.target.value)} />
            <i className="fas fa-search absolute right-3 top-3 text-gray-500"></i>
          </div>
        </div>
        <div className="h-[470px] overflow-y-scroll custom-scrollbar">
          <div className="grid grid-cols-2 gap-4">
            {currentData.map((item, i) => {
              return (
                <ItemTryFashionOn
                  classs={"max-h-[250px]"}
                  delay={-0}
                  offset={-999999}
                  id={item._id}
                  key={i}
                  name={item.name}
                  image={item.image}
                  new_price={item.new_price}
                  old_price={item.old_price}
                  isTry={true}
                  isCart={true}
                  fuctionOnClick={handleProductSelect}
                />
              );
            })}
          </div>
        </div>

      </section>
      <section className={`bg-white p-4 rounded-lg shadow-sm transition-all duration-200
        ${showInfor ? "col-span-1 lg:col-span-5" : "lg:col-span-8"}`}>
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold mb-4">Try Fashion</h2>
          <div className="flex items-center justify-between gap-1">
            <IoReload className="text-3xl font-bold p-2 bg-[#FF4141]/50 text-white rounded-md cursor-pointer hover:bg-[#ff4141]/60"
              title="Reload Camera" />
            <FaAngleDoubleLeft
              className={`group-hover:text-black text-white text-3xl 
             transition-all duration-400 p-2  bg-[#FF4141]/50 rounded-md cursor-pointer hover:bg-[#ff4141]/60
             ${showInfor ? "-rotate-180" : ""}`}
              onClick={() => setShowInfor(e => !e)}
              title="Show information"
            />
          </div>
        </div>
        <div className="relative bg-white rounded-2xl h-[560px] w-full overflow-hidden">
          <Camera setImageUpload={setImageUpload} />
        </div>
      </section>
      <section className={`${showInfor ? "col-span-1 lg:col-span-3 w-full" : "w-0 hidden"}
         bg-white p-4 rounded-lg shadow-sm col-span-1 lg:col-span-3 relative transition-all duration-0 overflow-hidden`}>
        <h2 className="text-lg font-semibold mb-4">Information</h2>
        <div className="flex flex-col gap-4">
          {product && product.image && product.image.length > 0 &&
            <div className="flex-1">
              <div className="font-poppins">Product selected</div>
              <div className="mx-auto w-1/2">
                <img className="rounded-lg" src={product.image[0]}></img>
              </div>
            </div>
          }
          {console.log(imageUpload)}
          {
            imageUpload &&
            <div className="flex-1">
              <div className="font-poppins">Image Upload</div>
              <div className="mx-auto w-64 h-64 overflow-hidden rounded-2xl">
                <img className="w-full h-full object-scale-down" src={URL.createObjectURL(imageUpload)}></img>
              </div>
            </div>
          }
        </div>
      </section>
    </div >
  );
};
