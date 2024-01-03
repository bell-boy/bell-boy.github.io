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

post0 = new BlogPost("college apps.", "why (and how) i spent a sizable chunk of my break writing a blog from scratch, and what i intend to do with it.", "today", "posts/00000000.html", "00000000")
post1 = new BlogPost("it's all just a linear map", "my plans for the new year, plus updates on projects relating to the blog and machine learning", "", "posts/00000001.html", "00000001")
let posts_container = document.querySelector("#posts-container")
posts_container.appendChild(post0.render())
posts_container.appendChild(post1.render())