import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

interface ProjectDescription {
  descriptionText: string;
  title: string;
  date: Date;

}

export default function ProjectCard(props: ProjectDescription) {
  const {descriptionText, title, date} = props
  return(
    <Card className="min-w-full">
      <CardHeader className="">
        <CardTitle>{title}</CardTitle>
        <CardDescription>{date.toDateString()}</CardDescription>
      </CardHeader>
      <CardContent>
        <p>{descriptionText}</p>
      </CardContent>
    </Card>    
  )
}