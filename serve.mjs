// little development server to get around cors when testing
import express from "express";

let app = express()
let port = 8080

app.use(express.static("./"))

app.listen(port, () => {
    console.log(`server running on port: ${port}`)
})