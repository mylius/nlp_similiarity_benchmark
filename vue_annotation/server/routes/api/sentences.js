
const express = require('express');
const mongodb = require('mongodb');

const router = express.Router();
const ObjectID = require('mongodb').ObjectID
// Get sentences

mongodb.MongoClient.connect('mongodb://localhost:27017/',  {
  useNewUrlParser: true
}, (err,client) =>{
  if(err) return console.log(err)
  sentences = client.db("annotationDB")
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
    const client = await mongodb.MongoClient.connect(
      'mongodb://localhost:27017/',
      {
        useNewUrlParser: true
      }
    );
    return client.db('annotationDB').collection('sentences');
  }


module.exports = router;