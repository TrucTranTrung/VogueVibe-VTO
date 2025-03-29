const express = require("express");
const app = express();
const mongoose = require("mongoose");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const path = require("path");
const cors = require("cors");
const port = process.env.PORT || 4000;

app.use(express.json());
app.use(cors());
mongoose.connect("your connection to database");

const firebase = require('./firebase')

const upload = multer({
  storage: multer.memoryStorage()
})

app.post('/upload', upload.array('product', 10), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).send('Error: No files found');
  }

  const uploadPromises = req.files.map((file) => {
    const fileName = `${Date.now().toString()}${path.extname(file.originalname)}`;

    const blob = firebase.bucket.file(fileName);
 
    const blobWriter = blob.createWriteStream({
      metadata: {
        contentType: file.mimetype,
      }, 
    });

    return new Promise((resolve, reject) => {
      blobWriter.on('error', (err) => {
        console.error(err);
        reject(err);
      });

      blobWriter.on('finish', async () => {
        // Thiết lập quyền truy cập công khai
        await blob.makePublic();
        
        const publicUrl = `https://storage.googleapis.com/${firebase.bucket.name}/${blob.name}`;
        console.log(blob.name)
        resolve(publicUrl);
      });

      blobWriter.end(file.buffer);
    });
  });

  try {
    const imageUrls = await Promise.all(uploadPromises);
    res.json({
      success: 1,
      image_url: imageUrls,
    });
  } catch (error) {
    console.error(error);
    res.status(500).send('Upload failed.');
  }
});


// MiddleWare to fetch user from token
const fetchuser = async (req, res, next) => {
  const token = req.header("auth-token");
  if (!token) {
    res.status(401).send({ errors: "Please authenticate using a valid token" });
  }
  try {
    const data = jwt.verify(token, "secret_ecom");
    req.user = data.user;
    next();
  } catch (error) {
    res.status(401).send({ errors: "Please authenticate using a valid token" });
  }
};


// Schema for creating user model
const Users = mongoose.model("Users", {
  name: { type: String },
  email: { type: String, unique: true },
  password: { type: String },
  cartData: { type: Object },
  date: { type: Date, default: Date.now() },
});


// Schema for creating Product
const Product = mongoose.model("Product", {
  name: { type: String, required: true },
  description: { type: String, required: true },
  image: [String],
  sex: { type: String, required: true },
  category: { type: String, require: true },
  new_price: { type: Number },
  old_price: { type: Number },
  date: { type: Date, default: Date.now },
  avilable: { type: Boolean, default: true },
});

// Schema for creating Category
const Category = mongoose.model("Categorys", {
  title: { type: String , required: true}
})


// add image 
app.get("/allimages", async (req, res) =>{
  let product = await Product.find({}, {image:1, _id:0});
  let images = product.map(item => (
    item.image
  ))
  res.send(images.flat())
})
 


// findbyImage
app.post("/findproductbyimg", async (req, res) =>{
  let data = req.body.images.map(item => item).join("|") ;
  let product2 = await Product.find({sex: req.body.gender, image: {$elemMatch: {$regex: data}}})
  res.send(product2)
})

 
app.get("/allimages/detect", async (req, res) =>{
  try {
    const { category: categories = [], gender = "" } = req.body; 
    const query = {};

    console.log("body", req.body);
    console.log("Categories", categories);
    console.log("gender", gender)

    if (categories.length > 0 && !gender) {
      query.category = { $in: categories.map((cat) => new RegExp(cat, "i")) };
    }
    if (gender && categories.length === 0) {
      query.sex = new RegExp(`^${gender}$`, "i"); 
    }
    if (categories.length > 0 && gender) {
      query.category = { $in: categories.map((cat) => new RegExp(cat, "i")) };
      query.sex = new RegExp(`^${gender}$`, "i");  
    }

    const data = await Product.find(query, { image: 1, _id: 0, sex:1 });

    console.log("query", query);
    
    const transformedData = data
      .flatMap((item) => item.image) 
      .map((image) => image.split("/").pop());

    // console.log("Transformed Data", transformedData);
    res.send(transformedData);
  } catch (error) {
    console.error("Error fetching images:", error);
    res.status(500).send({ error: "Internal Server Error" });
  }
})


// ROOT API Route For Testing
app.get("/", (req, res) => {
  res.send("Hi, API alrealy");
});


// Create an endpoint at ip/login for login the user and giving auth-token
app.post('/login', async (req, res) => {
  console.log("Login");
  let success = false;
  let user = await Users.findOne({ email: req.body.email });
  if (user) {
    const passCompare = req.body.password === user.password;
    if (passCompare) {
      const data = {
        user: {
          id: user.id
        }
      }
      success = true;
      const token = jwt.sign(data, 'secret_ecom');
      res.json({ success, token });
    }
    else {
      return res.status(400).json({ success: success, errors: "please try with correct email/password" })
    }
  }
  else {
    return res.status(400).json({ success: success, errors: "please try with correct email/password" })
  }
})


//Create an endpoint at ip/auth for regestring the user & sending auth-token
app.post('/signup', async (req, res) => {
  console.log("Sign Up");
  let success = false;
  let check = await Users.findOne({ email: req.body.email });
  if (check) {
    return res.status(400).json({ success: success, errors: "existing user found with this email" });
  }
  let cart = {};
  cart[1] = 0;
  const user = new Users({
    name: req.body.username,
    email: req.body.email,
    password: req.body.password,
    cartData: cart,
  });
  await user.save();
  const data = {
    user: {
      id: user.id
    }
  }

  const token = jwt.sign(data, 'secret_ecom');
  success = true;
  res.json({ success, token })
})


// endpoint for getting all products data
app.get("/allproducts/:type", async (req, res) => {
  let type = req.params.type.toUpperCase();
  let products = (type !== "ALL" ? await Product.find({ sex: type }) : await Product.find());
  console.log("All Products " + type);
  res.send(products);
});



// endpoint for getting latest products data
app.get("/newcollections", async (req, res) => {
  let products = await Product.find({});
  let arr = products.slice(0).slice(-8);
  console.log("New Collections");
  res.send(arr);
});

 
// endpoint for getting womens products data
app.get("/popularinwomen", async (req, res) => {
  let products = await Product.find({ sex: "FEMALE" });
  let arr = products.splice(0, 4);
  console.log("Popular In Women");
  res.send(arr);
});

// endpoint for getting womens products data
app.post("/relatedproducts", async (req, res) => {
  console.log("Related Products");
  const {category, sex} = req.body;
  const products = await Product.find({ category, sex });
  const arr = products.slice(0, 4);
  res.send(arr);
});


// Create an endpoint for saving the product in cart
app.post('/addtocart', fetchuser, async (req, res) => {
  console.log("Add Cart");
  let userData = await Users.findOne({ _id: req.user.id });
  if(isNaN(userData.cartData[req.body.itemId]) || userData.cartData[req.body.itemId] === null || userData.cartData[req.body.itemId] === undefined) 
    userData.cartData[req.body.itemId] = 1;
  else userData.cartData[req.body.itemId]++;
  await Users.findOneAndUpdate({ _id: req.user.id }, { cartData: userData.cartData });
  res.send("Added")
})


// Create an endpoint for removing the product in cart
app.post('/removefromcart', fetchuser, async (req, res) => {
  console.log("Remove Cart");
  let userData = await Users.findOne({ _id: req.user.id });
  if (userData.cartData[req.body.itemId] != 0) {
    userData.cartData[req.body.itemId] -= 1;
  }
  await Users.findOneAndUpdate({ _id: req.user.id }, { cartData: userData.cartData });
  res.send("Removed");
})


// Create an endpoint for getting cartdata of user
app.post('/getcart', fetchuser, async (req, res) => {
  console.log("Get Cart");
  let userData = await Users.findOne({ _id: req.user.id });
  res.json(userData.cartData);

})


// Create an endpoint for adding products using admin panel
app.post("/addproduct", async (req, res) => {
  const product = new Product({
    name: req.body.name,
    description: req.body.description,
    image: req.body.image,
    sex: req.body.sex,
    category: req.body.category,
    new_price: req.body.new_price,
    old_price: req.body.old_price,
  });
  await product.save();
  console.log("Saved");
  res.json({ success: true, data: product })
});




// Create an endpoint for removing products using admin panel
app.post("/removeproduct", async (req, res) => {
  await Product.findOneAndDelete({ id: req.body.id });
  console.log("Removed");
  res.json({ success: true, name: req.body.name })
});

// Starting Express Server
app.listen(port, (error) => {
  if (!error) console.log("Server Running on port " + port);
  else console.log("Error : ", error);
});




// backend cho try fashion



const Product2 = mongoose.model("Product2", {
  name: { type: String, required: false },
  description: { type: String, required: false },
  image: [String],
  sex: { type: String, required: false },
  category: { type: String, require: false },
  new_price: { type: Number },
  old_price: { type: Number },
  date: { type: Date, default: Date.now },
  avilable: { type: Boolean, default: true },
});


app.post("/addproduct2", async (req, res) => {
  const product = new Product2({
    name: req.body.name,
    description: req.body.description,
    image: req.body.image,
    sex: req.body.sex,
    category: req.body.category,
    new_price: req.body.new_price,
    old_price: req.body.old_price,
  });
  await product.save();
  console.log("Saved");
  res.json({ success: true, data: product })
});


app.get("/allproducts2/:type", async (req, res) => {
  let type = req.params.type.toUpperCase();
  let products = (type !== "ALL" ? await Product2.find({ sex: type }) : await Product2.find());
  console.log("All Products " + type);
  res.send(products);
});


app.post('/upload2', upload.array('product', 10), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).send('Error: No files found');
  }

  try {
    const uploadPromises = req.files.map((file) => {
      const fileName = `try_fashion/${file.originalname}`;
      const blob = firebase.bucket.file(fileName);
      const blobWriter = blob.createWriteStream({
        metadata: {
          contentType: file.mimetype,
        },
      });

      return new Promise((resolve, reject) => {
        blobWriter.on('error', (err) => {
          console.error(err);
          reject(err);
        });

        blobWriter.on('finish', async () => {
          await blob.makePublic();
          const publicUrl = `https://storage.googleapis.com/${firebase.bucket.name}/${fileName}`;
          console.log(publicUrl);
          resolve(publicUrl);
        });

        blobWriter.end(file.buffer);
      });
    });

    const uploadedFiles = await Promise.all(uploadPromises);
    res.status(200).json({ urls: uploadedFiles });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).send('Internal server error');
  }
});
