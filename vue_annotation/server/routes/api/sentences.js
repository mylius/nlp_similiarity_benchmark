
const express = require('express');
const mongodb = require('mongodb');
const Server = require('mongodb').Server;
const router = express.Router();
const ObjectID = require('mongodb').ObjectID
// Get sentences

var host = "localhost"
var port = 27017
var user = ""
var password = ""
// When using login data use mongodb.MongoClient.connect('mongodb://user:password@host:port/', [...] instead. !!!SAME FOR LINE 40!!!
mongodb.MongoClient.connect(new Server(host, port),  {
  useNewUrlParser: true
}, (err,client) =>{
  if(err) return console.log(err)
  //scheme is: sentences = client.db("dbname").collection("collection_name")
  sentences = client.db("annotations").collection("sentences")
})

router.get('/', async (req, res) => {
  const sentences = await loadSentencesCollection();
  res.send(await sentences.find({}).toArray());
});


// update sentence
router.put('/:id', (req, res) => {
  const _id = ObjectID(req.params.id)
  sentences.findOneAndUpdate({_id: _id},  { $set: {annotations: req.body.annotations}}, function(err, result){
    if(err){
      return res.status(500).json(err);
    }
    res.send(result)
  }); 
})

async function loadSentencesCollection() {
    const client = await mongodb.MongoClient.connect(new Server(host, port),
      {
        useNewUrlParser: true
      }
    );
    return client.db('annotations').collection('sentences');
  }


module.exports = router;