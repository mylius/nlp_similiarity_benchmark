import axios from 'axios';


const url = 'http://localhost:5000/api/sentences/';

class DataService {
    /**
     * Reads all sentences from mongo and maps them to data.
     */
    static getSentences(){
        return new Promise(async (resolve,reject)=>{
            try{
                const res = await axios.get(url);
                const data = res.data;
                resolve(
                    data.map(sentence => ({
                        ...sentence,
                        id: new String(sentence._id),
                        sent: new String(sentence.sent),
                        annotation: new Object(sentence.annotations)
                    }))
                );
            } catch(err){
                reject(err);
            }
        })
    }
    /**
     * Updates the entry of the annotated sentence in mongo.
     * @param {*} params The parameters of the annotated sentence
     */
    static  updateSentence (params) {
        return axios.put(url + params.id, params)
    }
}

export default DataService;