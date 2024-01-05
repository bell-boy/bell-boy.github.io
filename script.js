class BlogPost {
    constructor(title, description, date, link, id){
        this.title = title
        this.description = description
        this.date = date
        this.link = link
        this.id = id
    }   

    render() {
        let element = document.createElement("div")
        element.classList.add(..."card d-flex flex-column border rounded p-3".split(" "))

        let title_element = document.createElement("h2")
        title_element.classList.add(..."card-title m-0".split(" "))
        title_element.innerText = this.title

        let id_element = document.createElement("p")
        id_element.classList.add(..."small m-0".split(" "))
        id_element.innerText = this.id

        let description_element = document.createElement("p")
        description_element.innerText = this.description

        let link_element = document.createElement("a")
        link_element.classList.add("stretched-link")
        link_element.href = this.link

        element.appendChild(title_element)
        element.appendChild(id_element)
        element.appendChild(description_element)
        element.appendChild(link_element)
        return element
    }
}

let posts_container = document.querySelector("#posts-container")
fetch("./postmetadata.json").then((res) => {
    if(!res.ok) {
        throw new Error(res.status)
    }
    return res.json()
}).then((val) => {
    val.posts.forEach(element => {
        posts_container.appendChild((new BlogPost(element.title, element.description, "", element.path, element.id)).render())
    });
})