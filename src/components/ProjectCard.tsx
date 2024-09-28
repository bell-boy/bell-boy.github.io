import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

import { ProjectDescription } from "@/data/projects"


export default function ProjectCard(props: ProjectDescription) {
  const {name, date, description, link} = props
  return(
    <a href={link} target="_blank" className="col-span-1">
      <Card className="min-w-full break-words">
        <CardHeader className="">
          <CardTitle>{name}</CardTitle>
          <CardDescription>{(new Date(date)).toDateString()}</CardDescription>
        </CardHeader>
        <CardContent>
          <p>{description}</p>
        </CardContent>
      </Card>    
    </a>
  )
}