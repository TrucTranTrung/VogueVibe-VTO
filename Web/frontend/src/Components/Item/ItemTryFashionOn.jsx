import React, { memo, useContext } from "react";
import "./Item.css";
import { Link } from "react-router-dom";
import { ShopContext } from '../../Context/ShopContext'

const ItemTryFashionOn = memo((props) => {
  const handleScrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  const handleLinkClick = (Event) =>{
    if(props.isTry){
      props.fuctionOnClick(props.id)
      Event.preventDefault();
      return;
    }
    handleScrollToTop();
  }
  return (
    <div
      data-aos="fade-up"
      data-aos-delay={props.delay}
      data-aos-offset={props.offset}
      className={`item group shadow-xl rounded-xl p-2 cursor-pointer ${props.classs}`}
    >
      <Link className="flex flex-col h-full duration-200 transition-all" to={"/product/" + props.id} onClick={(e) => handleLinkClick(e)}>
        <div className="rounded-md overflow-hidden h-full w-full relative group">
          <img
            className="w-full h-full object-cover group-hover:scale-110 transition-all duration-200"
            onClick={window.scrollTo(0, 0)}
            src={props.image[0]}
            alt="products"
          />
          {props.isTry &&
            <div className="absolute bottom-0 left-0 right-0 group-hover:h-full group-hover:visible invisible  h-0 transition-all duration-300 bg-gray-400/40 flex items-center justify-center">
              <p className="text-center  text-white">Try it now</p>
            </div>}
        </div>
      </Link>
    </div>
  );
});

export default ItemTryFashionOn;
